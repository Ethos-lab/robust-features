import dill
import tqdm
import argparse
import torch as ch

from robustness import model_utils, datasets, cifar_models
from robustness.attacker import AttackerModel

# Constants
DATA_SHAPE = 32 if DATA == 'CIFAR' else 224 # Image size (fixed for dataset)


def get_args():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument('--dataset', type=str, choices=["CIFAR"], default='CIFAR', help='dataset')
    parser.add_argument('--dataroot', type=str, default='/media/big_hdd/data/CIFAR10', help='path to dataset if you need diff from default')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--load-path', type=str, default='')
    parser.add_argument('--model-type', type=str, choices=['attacker_model', 'nn_module'], default='nn_module')
    parser.add_argument('--batch-size', type=int, default=50, help='initial batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='for dataloader')

    args = parser.parse_args()

    return args


def test(amodel, test_loader, make_adv, attack_kwargs=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader))
    for im, lbl in test_loader:
        im, lbl = im.cuda(), lbl.cuda()
        if make_adv:
            assert attack_kwargs != None:
            _, im_adv = amodel(im, lbl, make_adv=True, **attack_kwargs)
            pred, _ = amodel(im_adv)
        else:
            pred, _ = amodel(im, make_adv=False)
        label_pred = ch.argmax(pred, dim=1)
        correct += (label_pred == lbl).sum().item()
        total += im.shape[0]
        pbar.update(1)
    pbar.close()

    return 100*correct/total


def main():
    # Get arguments
    args = get_args()

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # Load dataset
    dataset_function = getattr(datasets, args.dataset)
    dataset = dataset_function(args.dataroot)
    _, test_loader = dataset.make_loaders(workers=args.num_workers, 
                                          batch_size=args.batch_size, 
                                          data_aug=False)
    data_iterator = enumerate(test_loader)

    model = cifar_models.__dict__[args.model](num_classes=10).cuda()

    ckpt_path = f'./checkpoints/{args.load_path}/model.pt'
    if args.model_type == 'nn_module':
        ckpt = ch.load(ckpt_path)
        sd = ckpt['state_dict']
        model.load_state_dict(sd)
        print(f"Successfully loaded checkpoint from epoch: {ckpt['epoch']}")

    # wrap nn.module object into AttackerModel to make it compatible with robustness toolbox
    amodel = AttackerModel(model, dataset).cuda()
    amodel.eval()

    if args.model_type == 'attacker_model':
        checkpoint = ch.load(ckpt_path, pickle_module=dill)
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        amodel.load_state_dict(sd)

    # testing begins

    # Natural accuracy
    acc = test(amodel, test_loader, make_adv=False, attack_kwargs={})
    print(f'Natural accuracy: {:.4f}%')

    # PGD (eps = 0.25)
    attack_kwargs = {
        'constraint':'2', # use L2-PGD
        'eps': 0.25, # L2 radius around original image
        'step_size': 0.03125, #0.25: 0.03125, 0.5: 0.0625, 1: 0.125,
        'iterations': 20,
        'do_tqdm': False
    }
    acc = test(amodel, test_loader, make_adv=True, attack_kwargs=attack_kwards)
    print(f'Adversarial accuracy (0.25): {acc:.4f}%')

    # PGD (eps = 0.5)
    attack_kwargs = {
        'constraint':'2', # use L2-PGD
        'eps': 0.5,
        'step_size': 0.0625,
        'iterations': 20,
        'do_tqdm': False
    }
    acc = test(amodel, test_loader, make_adv=True, attack_kwargs=attack_kwards)
    print(f'Adversarial accuracy (0.5): {acc:.4f}%')

    # PGD (eps = 1.0)
    attack_kwargs = {
        'constraint':'2', # use L2-PGD
        'eps': 1.0,
        'step_size': 0.125,
        'iterations': 20,
        'do_tqdm': False
    }
    acc = test(amodel, test_loader, make_adv=True, attack_kwargs=attack_kwards)
    print(f'Adversarial accuracy (1.0): {acc:.4f}%')


if __name__ == '__main__':
    main()

