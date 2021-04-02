import os
import sys
import tqdm
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from robustness import model_utils, datasets, cifar_models
from robustness.tools import helpers, folder


def get_args():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument('--dataset', type=str, choices=["CIFAR"], default='CIFAR', help='dataset')
    parser.add_argument('--dataroot', type=str, default='/media/big_hdd/data/CIFAR10', help='path to dataset if you need diff from default')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--rob-model-path', type=str, default='cifar_resnet50_l2_1_0')
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')

    # training hyperparams
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='initial batch size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--milestones', type=str, default='65,90')
    parser.add_argument('--weight-decay', type=float, default=0.00005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--num-workers', type=int, default=4, help='for dataloader')
    parser.add_argument('--xent-weight', type=float, default=1.)
    parser.add_argument('--l2-weight', type=float, default=1.)
    parser.add_argument('--normalize-l2-loss', action='store_true')
    parser.add_argument('--load-std-data', action='store_true')

    # testing hyperparams
    parser.add_argument('--val-batch-size', type=int, default=128, help='initial batch size')

    args = parser.parse_args()

    return args


class ModelwithNormalization(torch.nn.Module):
    def __init__(self, model, dataset):
        super(ModelwithNormalization, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self,  inp, with_latent=False, fake_relu=False, no_relu=False):
        normalized_inp = self.normalizer(inp)

        if no_relu and (not with_latent):
            print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
        if no_relu and fake_relu:
            raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

        output = self.model(normalized_inp, with_latent=with_latent,
                            fake_relu=fake_relu, no_relu=no_relu)

        return output


def train(args, net, f_rob, trainloader, valloader, criterion, optimizer, scheduler, logger, start_epoch, best_acc):
    logger.info('Started Training !!!')

    for epoch in range(start_epoch, args.epochs):  # loop over the dataset multiple times
        print(f'Epoch: {epoch}/{args.epochs}')
        net.train()

        running_loss = 0.0
        running_xent = 0.0
        running_fl = 0.0
        total = 0.0
        correct = 0.0
        pbar = tqdm.tqdm(total=len(trainloader))
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            #TODO: forwarding through net before f_rob causes GPU to run out of memory. Why?
            if args.l2_weight != 0:
                (_, rep_rob), _ = f_rob(images, with_latent=True)
                outputs, rep_std = net(images, with_latent=True)
                if args.normalize_l2_loss:
                    loss_feat = torch.mean(torch.div(
                                    torch.norm(rep_std - rep_rob.detach(), dim=1),
                                    torch.norm(rep_rob.detach(), dim=1)
                                ))
                else:
                    loss_feat = torch.mean(torch.norm(rep_std - rep_rob.detach(), dim=1))
            else:
                outputs, rep_std = net(images, with_latent=True)
                loss_feat = torch.tensor(0)

            loss_xent = criterion(outputs, labels)

            loss = args.xent_weight*loss_xent + args.l2_weight*loss_feat
            loss.backward()
            optimizer.step()

            running_xent += loss_xent.item()
            running_fl += loss_feat.item()
            running_loss += loss.item()
            #print(running_loss/(i+1), running_xent/(i+1), running_fl/(i+1))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.update(1)
        pbar.close()

        # evaluate at end of each epoch
        train_acc = 100 * correct / total
        logger.info(f'Train | '+\
                    f'loss: {running_loss/len(trainloader):.4f} ({running_xent/len(trainloader):.4f}, {running_fl/len(trainloader):.4f}), '+\
                    f'accuracy: {train_acc:.4f}%')

        val_acc = test(net, valloader)
        logger.info(f'Val | accuracy: {val_acc:.4f}%')

        if val_acc > best_acc:
            logger.info('Saving model !!!')
            best_acc = val_acc
            # save model
            torch.save({
                    'state_dict': net.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                },
                os.path.join(args.save_dir, 'model.pt')
            )

        scheduler.step()

    logger.info('Finished Training !!!')


def test(net, valloader):
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # forward pass
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    return acc


def main():
    # Setting up logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    args = get_args()

    if args.save_dir == '':
        args.save_dir = f'./checkpoints/{args.dataset}-feat_loss=l2-teacher={args.rob_model_path}-student={args.model}-loss={args.xent_weight}xent+{args.l2_weight}l2_loss'
        if args.exp_name != '':
           args.save_dir += f'-{args.exp_name}'
    args.rob_model_arch = args.rob_model_path.split('_')[1]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    logger.addHandler(logging.FileHandler(os.path.join(args.save_dir, 'log.txt')))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # Load dataset
    dataset_function = getattr(datasets, args.dataset)
    if args.load_std_data:
        dataset = dataset_function(args.dataroot)
        train_loader, val_loader = dataset.make_loaders(workers=args.num_workers,
                                                         batch_size=args.batch_size,
                                                         val_batch_size=args.val_batch_size,
                                                         data_aug=True,
                                                         shuffle_train=True,
                                                         shuffle_val=False)
    else:
        dataset = dataset_function('') #Only use this for accessing mean and std

        # Loading robust dataset
        TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.25,.25,.25),
                    transforms.RandomRotation(2),
                    transforms.ToTensor(),
                ])

        TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor()
        ])

        train_data = torch.cat(torch.load(os.path.join(args.dataroot, f"CIFAR_ims")))
        train_labels = torch.cat(torch.load(os.path.join(args.dataroot, f"CIFAR_lab")))
        train_set = folder.TensorDataset(train_data, train_labels, transform=TRAIN_TRANSFORMS_DEFAULT(32))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        val_set = torchvision.datasets.CIFAR10(root='/media/big_hdd/data/CIFAR10', train=False,
                                               transform=TEST_TRANSFORMS_DEFAULT(32))
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Prepare standard and robust model
    if args.l2_weight != 0:
        model_kwargs = {
            'arch': args.rob_model_arch,
            'dataset': dataset,
            'resume_path': f'./checkpoints/{args.rob_model_path}/model.pt'
        }
        rob_model, _ = model_utils.make_and_restore_model(**model_kwargs)
        rob_model.eval()
    else:
        rob_model = None

    model = cifar_models.__dict__[args.model](num_classes=10)
    # Add normalizer
    norm_model = ModelwithNormalization(model, dataset).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Load pre-trained model
    if os.path.exists(os.path.join(args.save_dir, 'model.pt')):
        ckpt = torch.load(os.path.join(args.save_dir, 'model.pt'))
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']

        logger.info(f'Loading pre-trained model for finetuning !!!')
    else:
        start_epoch = 0
        best_acc = 0

    scheduler = None
    if args.milestones != '':
        milestones = [int(x) for x in args.milestones.split(',')]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    train(args, norm_model, rob_model, train_loader, val_loader, criterion, optimizer, scheduler, logger, start_epoch, best_acc)


if __name__ == '__main__':
    main()
