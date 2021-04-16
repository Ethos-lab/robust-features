import os
import sys
import tqdm
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from robustness import cifar_models, imagenet_models
from robustness.tools import helpers, folder
from robustness.datasets import DATASETS

from utils import load_checkpoint, ModelwithInputNormalization
from attacks import PGD_l2
from logfile_to_wandb import upload_to_wandb


def get_args():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"],
                        default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/CIFAR10",
                        help="path to dataset")
    parser.add_argument("--rob-dataroot", type=str, default="./datasets/d_robust_CIFAR",
                        help="path to robust dataset")
    parser.add_argument("--load-std-data", action="store_true",
                        help="loads D if true, else loads D_r")
    parser.add_argument("--student-arch", type=str, default="resnet50")
    parser.add_argument("--teacher-load-path", type=str,
                        default="/path/to/teacher/checkpoint.pt.best")
    parser.add_argument("--exp-name", type=str, default="",
                        help="additional description for exp to add to save_dir name")
    parser.add_argument("--save-dir", type=str, default="",
                        help="overwrites automatically created save_dir")
    parser.add_argument("--resume", type=str, default="",
                        help="resume training from the checkpoint saved at this path")
    # training hyperparams
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="initial batch size")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="gamma of optim.lr_scheduler.StepLR, decay of lr")
    parser.add_argument("--milestones", type=str, default="65,90")
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    parser.add_argument("--xent-weight", type=float, default=1.)
    parser.add_argument("--l2-weight", type=float, default=1.)
    # testing hyperparams
    parser.add_argument("--val-batch-size", type=int, default=128, help="initial batch size")
    # wandb args
    parser.add_argument("--upload-to-wandb", action="store_true",
                        help="upload log.txt to wandb for visualization")
    parser.add_argument("--wandb-username", type=str, default="pratik18v")
    parser.add_argument("--wandb-project-name", type=str, default="robust-features")
    parser.add_argument("--wandb-exp-name", type=str, default="")

    args = parser.parse_args()

    return args


def train(args, s_model, t_model, trainloader, valloader, criterion, optimizer, scheduler, logger,
          start_epoch, best_acc, r_time_avg, attack=None):

    num_batches = len(trainloader)
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch: {epoch}/{args.epochs} (@ {r_time_avg:.4f} secs/epoch)")
        s_model.train()

        # Init counters
        r_loss = 0.0
        r_xent = 0.0
        r_fl = 0.0
        total = 0.0
        correct = 0.0

        start_t = time.time()
        pbar = tqdm.tqdm(total=num_batches)
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            if args.l2_weight != 0:
                outputs, rep_std = s_model(images, with_latent=True)
                with torch.no_grad():
                    _, rep_rob = t_model(images, with_latent=True)
                loss_feat = torch.mean(torch.div(
                                torch.norm(rep_std - rep_rob.detach(), dim=1),
                                torch.norm(rep_rob.detach(), dim=1)
                            ))
            else:
                outputs, rep_std = s_model(images, with_latent=True)
                loss_feat = torch.tensor(0)

            loss_xent = criterion(outputs, labels)

            loss = args.xent_weight*loss_xent + args.l2_weight*loss_feat
            loss.backward()
            optimizer.step()

            r_xent += loss_xent.item()
            r_fl += loss_feat.item()
            r_loss += loss.item()

            pbar.set_description(
                f"Train | loss: {r_loss/(i+1):.4f} ({r_xent/(i+1):.4f}, {r_fl/(i+1):.4f})")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.update(1)
        pbar.close()

        # update running time average
        curr_time = time.time() - start_t
        if epoch == 0:
            r_time_avg = curr_time
        else:
            r_time_avg = r_time_avg + ((curr_time - r_time_avg) / epoch)

        # evaluate at end of each epoch
        train_acc = 100 * correct / total
        logger.info(f"Train | "+\
            f"loss: {r_loss/num_batches:.4f} ({r_xent/num_batches:.4f}, {r_fl/num_batches:.4f}), "+\
            f"accuracy: {train_acc:.4f}%")

        val_nat_acc, val_adv_acc = test(s_model, valloader, attack)
        if attack:
            logger.info(f"Val | std accuracy: {val_nat_acc:.4f}%, adv accuracy: {val_adv_acc:.4f}%")
            val_acc = val_adv_acc
        else:
            logger.info(f"Val | std accuracy: {val_nat_acc:.4f}%")
            val_acc = val_nat_acc

        if val_acc > best_acc:
            logger.info("Saving model !!!")
            best_acc = val_acc
            torch.save({
                    "state_dict": s_model.net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "val_nat_acc": val_nat_acc,
                    "val_adv_acc": val_adv_acc,
                    "time_per_epoch": r_time_avg
                },
                os.path.join(args.save_dir, "checkpoint.pt.best")
            )

        scheduler.step()

    logger.info(f"Finished Training (@ {r_time_avg:.4f} secs/epoch)!!!")


def test(model, valloader, attack):
    model.eval()

    correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if attack:
                adv_images = attack(model, images, labels)
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                adv_correct += (predicted == labels).sum().item()

    nat_acc = 100 * correct / total
    adv_acc = 100 * adv_correct / total

    return nat_acc, adv_acc


def main():
    # Setting up logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    args = get_args()

    if args.save_dir == "":
        if args.l2_weight == 0:
            assert args.xent_weight == 1.0
            args.save_dir = f"./checkpoints/{args.dataset}_{args.student_arch}"
        else:
            args.teacher_arch = args.teacher_load_path.split("_")[1]
            args.save_dir = f"./checkpoints/{args.dataset}-feat_loss=l2-teacher={args.teacher_arch}-student={args.student_arch}-xent_weight={args.xent_weight}-l2_weight={args.l2_weight}"

        if args.exp_name != "":
           args.save_dir += f"-{args.exp_name}"
    else:
        if args.l2_weight == 0:
            assert args.xent_weight == 1.0
        else:
            args.teacher_arch = args.teacher_load_path.split("_")[1]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    logfile_path = os.path.join(args.save_dir, "log.txt")
    logger.addHandler(logging.FileHandler(logfile_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    dataset_function = DATASETS[args.dataset]
    dataset = dataset_function(args.dataroot)
    if args.load_std_data:
        # Load standard dataset (D)
        train_loader, val_loader = dataset.make_loaders(workers=args.num_workers,
                                                        batch_size=args.batch_size,
                                                        val_batch_size=args.val_batch_size,
                                                        data_aug=True,
                                                        shuffle_train=True,
                                                        shuffle_val=False)
    else:
        assert args.dataset == "cifar", "Robust data loading only supported for CIFAR-10 !!!"
        # Load robust dataset (D_r)
        TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.25,.25,.25),
                    transforms.RandomRotation(2),
                    transforms.ToTensor()])
        train_data = torch.cat(torch.load(os.path.join(args.rob_dataroot, "CIFAR_ims"))).cpu()
        train_labels = torch.cat(torch.load(os.path.join(args.rob_dataroot, "CIFAR_lab"))).cpu()
        train_set = folder.TensorDataset(train_data, train_labels, transform=TRAIN_TRANSFORMS_DEFAULT(32))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        #NOTE: Need path to standard data to load val set
        _,val_loader = dataset.make_loaders(only_val=True,
                                            batch_size=args.val_batch_size,
                                            shuffle_val=False,
                                            workers=args.num_workers)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)

    # Prepare teacher model
    if args.l2_weight != 0:
        if args.dataset == "cifar":
            t_model = cifar_models.__dict__[args.teacher_arch](num_classes=10)
        else:
            t_model = imagenet_models.__dict__[args.teacher_arch](num_classes=9)

        ckpt = load_checkpoint(args.teacher_load_path)
        t_model.load_state_dict(ckpt["model_sd"])
        print(f"Successfully loaded teacher checkpoint from epoch: {ckpt['load_epoch']} !!!")
        del ckpt

        t_model = ModelwithInputNormalization(t_model, dataset.mean, dataset.std).cuda()
        t_model.eval()
    else:
        t_model = None

    # Prepare student model
    if args.dataset == "cifar":
        s_model = cifar_models.__dict__[args.student_arch](num_classes=10)
    else:
        s_model = imagenet_models.__dict__[args.student_arch](num_classes=9)
    s_model = ModelwithInputNormalization(s_model, dataset.mean, dataset.std).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Resume student training
    if args.resume != "":
        assert os.path.exists(args.resume), f"Invalid load path: {args.resume}"
        ckpt = load_checkpoint(args.resume, mode="train")
        s_model.net.load_state_dict(ckpt["model_sd"])
        optimizer.load_state_dict(ckpt["optim_sd"])
        start_epoch = ckpt["load_epoch"] + 1
        best_acc = ckpt["best_acc"]
        r_time_avg = ckpt["r_time_avg"]
        del ckpt

        logger.info(f"Resuming training from epoch {start_epoch} !!!")
    else:
        start_epoch = 0
        best_acc = 0
        r_time_avg = 0

    scheduler = None
    if args.milestones != "":
        milestones = [int(x) for x in args.milestones.split(",")]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    # Init attack for eval
    #TODO: set attack params using args
    #attack = PGD_l2(epsilon=1., num_steps=20, step_size=0.125).cuda()
    attack = None

    logger.info("Started Training !!!")
    train(args, s_model, t_model, train_loader, val_loader, criterion, optimizer, scheduler,
          logger, start_epoch, best_acc, r_time_avg, attack)

    # Upload logfile to wandb
    if args.upload_to_wandb:
        upload_to_wandb(logfile_path, args.wandb_project_name, args.wandb_username, args.wandb_exp_name)


if __name__ == "__main__":
    main()
