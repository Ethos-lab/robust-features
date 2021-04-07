import dill
import tqdm
import argparse
import torch as ch

from robustness import model_utils, datasets, cifar_models, imagenet_models
from robustness.attacker import AttackerModel
from robustness.datasets import DATASETS
from robustness.tools import helpers


'''
Sample PGD hyperparameters

# CIFAR-10
## PGD (eps = 0.25)
attack_kwargs = {
    "constraint":"2", # use L2-PGD
    "eps": 0.25, # L2 radius around original image
    "step_size": 0.03125,
    "iterations": 20,
    "random_start": True,
    "do_tqdm": False
}

## PGD (eps = 0.5)
attack_kwargs = {
    "constraint":"2",
    "eps": 0.5,
    "step_size": 0.0625,
    "iterations": 20,
    "random_start": True,
    "do_tqdm": False
}

## PGD (eps = 1.0)
attack_kwargs = {
    "constraint":"2",
    "eps": 1.0,
    "step_size": 0.125,
    "iterations": 20,
    "random_start": True,
    "do_tqdm": False
}

# Restricted Imagenet
## PGD (eps = 0.5)
attack_kwargs = {
    "constraint":"2",
    "eps": 0.5,
    "step_size": 0.0625,
    "iterations": 20,
    "random_start": True,
    "do_tqdm": False
}

## PGD (eps = 3.0)
attack_kwargs = {
    "constraint":"2",
    "eps": 3.0,
    "step_size": 0.375,
    "iterations": 20,
    "random_start": True,
    "do_tqdm": False
}

General rule of thumb:
  - step_size = 2.5*eps/iterations
'''

def get_args():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"], default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/CIFAR10", help="path to dataset")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams (no default values, pass every time)
    parser.add_argument("--eps", type=float, required=True, help="perturbation budget")
    parser.add_argument("--pgd-iters", type=int, required=True, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, required=True, help="perturb step size")
    parser.add_argument("--constraint", type=str, choices=["2", "inf"], required=True, help="PGD attack constraint metric")
    parser.add_argument("--no_rand_init", action="store_true", help="random initialization for PGD")

    args = parser.parse_args()

    return args


def test(amodel, test_loader, make_adv, attack_kwargs=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader))
    for im, lbl in test_loader:
        im, lbl = im.cuda(), lbl.cuda()
        if make_adv:
            assert attack_kwargs != None
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
        print(f"Argument {arg}: {value}")

    # Load dataset
    dataset_function = DATASETS[args.dataset]
    dataset = dataset_function(args.dataroot)
     _, test_loader = dataset.make_loaders(only_val=True,
                                           val_batch_size=args.batch_size,
                                           shuffle_val=False,
                                           workers=args.num_workers
                                           )
    test_loader = helpers.DataPrefetcher(test_loader)

    if args.dataset == "cifar":
        model = cifar_models.__dict__[args.student_arch](num_classes=10)
    else:
        model = imagenet_models.__dict__[args.student_arch](num_classes=9)

    # Loading model checkpoint
    checkpoint = ch.load(args.load_path, pickle_module=dill)
    state_dict_path = "model"
    if not ("model" in checkpoint):
        state_dict_path = "state_dict"

    sd = checkpoint[state_dict_path]
    sd = {k.replace("module.model.", ""):v for k,v in sd.items() if ("attacker" not in k) and ("normalizer" not in k)}
    model.load_state_dict(sd)
    print(f"Successfully loaded checkpoint from epoch: {checkpoint['epoch']}")

    # clear out memory
    del checkpoint

    # wrap nn.module object into AttackerModel to make it compatible with robustness toolbox
    amodel = AttackerModel(model, dataset).cuda()
    amodel.eval()

    # Testing begins

    # Natural accuracy
    acc = test(amodel, test_loader, make_adv=False)
    print(f"Natural accuracy: {acc:.4f}%")

    # PGD eval
    attack_kwargs = {
        "constraint":args.constraint, # use L2-PGD
        "eps": args.eps, # L2 radius around original image
        "step_size": args.step_size,
        "iterations": args.pgd_iters,
        "random_start": not args.no_rand_init,
        "do_tqdm": False
    }
    acc = test(amodel, test_loader, make_adv=True, attack_kwargs=attack_kwargs)
    print(f"Adversarial accuracy (0.25): {acc:.4f}%")


if __name__ == "__main__":
    main()

