import dill
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn

from robustness import model_utils, datasets, cifar_models
from robustness.tools import helpers

from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier


def get_args():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument("--dataset", type=str, choices=["CIFAR"], default="CIFAR", help="dataset")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/CIFAR10", help="path to dataset if you need diff from default")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")

    args = parser.parse_args()

    return args


def test(model, test_loader, attack=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader))
    for im, lbl in test_loader:
        im, lbl = im.numpy(), lbl.numpy()
        if attack:
            im_adv = attack.generate(im, lbl)
            pred = model.predict(im_adv)
        else:
            pred = model.predict(im)
        label_pred = np.argmax(pred, axis=1)
        correct += (label_pred == lbl).sum()
        total += im.shape[0]
        pbar.update(1)
    pbar.close()

    return 100*correct/total


class ModelwithInputNormalization(nn.Module):
    def __init__(self, model, mean, std):
        super(ModelwithInputNormalization, self).__init__()
        self.normalizer = helpers.InputNormalize(mean, std)
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


def main():
    # Get arguments
    args = get_args()

    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # Load dataset
    dataset_function = getattr(datasets, args.dataset)
    dataset = dataset_function(args.dataroot)
    _, test_loader = dataset.make_loaders(workers=args.num_workers, 
                                          batch_size=args.batch_size, 
                                          data_aug=False,
                                          only_val=True,
                                          shuffle_val=False)

    model = cifar_models.__dict__[args.model](num_classes=10).cuda()

    checkpoint = torch.load(args.load_path, pickle_module=dill)
    state_dict_path = "model"
    if not ("model" in checkpoint):
        state_dict_path = "state_dict"

    sd = checkpoint[state_dict_path]
    sd = {k.replace("module.model.", ""):v for k,v in sd.items() if ("attacker" not in k) and ("normalizer" not in k)}
    model.load_state_dict(sd)
    print(f"Successfully loaded checkpoint from epoch: {checkpoint['epoch']}")

    model = ModelwithInputNormalization(model, dataset.mean, dataset.std)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    art_model = PyTorchClassifier(model=model, loss=criterion, input_shape=(3,32,32), nb_classes=10, clip_values=(0,1))


    # Testing begins

    # Natural accuracy
    acc = test(art_model, test_loader)
    print(f"Natural accuracy: {acc:.4f}%")

    # PGD (eps = 0.25)
    attack_kwargs = {
        "norm":2, # use L2-PGD
        "eps": 0.25,
        "eps_step": 0.03125,  #0.25: 0.03125, 0.5: 0.0625, 1: 0.125
        "max_iter": 20,
        "targeted": False,
        "num_random_init": 1,
        "batch_size": args.batch_size,
        "verbose": False
    }
    adversary = ProjectedGradientDescentPyTorch(art_model, **attack_kwargs)
    acc = test(art_model, test_loader, adversary)
    print(f"Adversarial accuracy (0.25): {acc:.4f}%")

    # PGD (eps = 0.5)
    attack_kwargs = {
        "norm":2, # use L2-PGD
        "eps": 0.5,
        "eps_step": 0.0625,
        "max_iter": 20,
        "targeted": False,
        "num_random_init": 1,
        "batch_size": args.batch_size,
        "verbose": False
    }
    adversary = ProjectedGradientDescentPyTorch(art_model, **attack_kwargs)
    acc = test(art_model, test_loader, adversary)
    print(f"Adversarial accuracy (0.5): {acc:.4f}%")

    # PGD (eps = 1.0)
    attack_kwargs = {
        "norm":2, # use L2-PGD
        "eps": 1.0,
        "eps_step": 0.125,
        "max_iter": 20,
        "targeted": False,
        "num_random_init": 1,
        "batch_size": args.batch_size,
        "verbose": False
    }
    adversary = ProjectedGradientDescentPyTorch(art_model, **attack_kwargs)
    acc = test(art_model, test_loader, adversary)
    print(f"Adversarial accuracy (1.0): {acc:.4f}%")


if __name__ == "__main__":
    main()

