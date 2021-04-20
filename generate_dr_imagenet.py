import os
import math
import tqdm
import argparse
import numpy as np
from PIL import Image

import torch as ch

from robustness import model_utils
from robustness.datasets import DATASETS
from robustness.tools import helpers
import robustness.data_augmentation as da

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def get_args():
    parser = argparse.ArgumentParser(description="Robust data generation script.")
    parser.add_argument("--dataset", type=str, choices=["restricted_imagenet"],
                        default="restricted_imagenet", help="dataset")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/imagenet",
                        help="path to dataset")
    parser.add_argument("--model-dir", type=str, default="rimagenet_alexnet_l2_3_0")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams (no default values, pass every time)
    parser.add_argument("--iterations", type=int, default=2000, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=1.0, help="perturb step size")
    parser.add_argument("--constraint", type=str, choices=["2", "inf"], default="2",
                        help="PGD attack constraint metric")

    args = parser.parse_args()

    return args


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def main():

    # Get arguments
    args = get_args()

    # NOTE: Model arch inferred from arg: model_dir.
    args.model = args.model_dir.split("_")[1]

    args.save_dir = f"{args.dataroot}_constructed/{args.model_dir}/train"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # Load dataset
    dataset_function = DATASETS[args.dataset]
    dataset = dataset_function(args.dataroot)
    train_loader, _ = dataset.make_loaders(data_aug=False,
                                           batch_size=args.batch_size,
                                           workers=args.num_workers)
    train_loader = helpers.DataPrefetcher(train_loader)

    # Load model
    model_kwargs = {
        "arch": args.model,
        "dataset": dataset,
        "resume_path": f"./checkpoints/{args.model_dir}/checkpoint.pt.best"
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()

    # Custom loss for inversion
    def inversion_loss(model, inp, targ):
        _, rep = model(inp, with_latent=True)
        loss = ch.norm(rep - targ, dim=1)
        return loss, None

    # PGD parameters
    kwargs = {
        "custom_loss": inversion_loss,
        "constraint": args.constraint,
        "eps": 1e3, #no restriction on how much to change image
        "step_size": args.step_size,
        "iterations": args.iterations,
        "do_tqdm": False,
        "targeted": True,
        "use_best": True
    }

    # Generate list of all image paths
    print("==> Indexing files")
    all_paths = []
    class_keys = os.listdir(os.path.join(args.dataroot, 'train'))
    pbar = tqdm.tqdm(total=len(class_keys))
    for target in class_keys:
        d = os.path.join(args.dataroot, 'train', target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for ii, fname in enumerate(sorted(fnames)):
                pbar.set_description(f"Progress: {ii+1}/{len(fnames)}")
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    all_paths.append(path)

        pbar.update(1)
    pbar.close()
    print(f"Total images to process: {len(all_paths)}")

    print("==> Preparing robust dataset")
    num_batches = int(math.ceil(len(all_paths) / args.batch_size))
    pbar = tqdm.tqdm(total=num_batches)
    for ibatch in range(num_batches):
        bstart = ibatch * args.batch_size
        bend = min(bstart + args.batch_size, len(all_paths))

        # read images in batches using the list of all paths
        paths = all_paths[bstart:bend]
        im_batch = []
        for path in paths:
            with open(path, 'rb') as f:
                im = Image.open(f)
                im = im.convert('RGB')
            im = da.TEST_TRANSFORMS_IMAGENET(im)
            im_batch.append(im)
        im_batch = ch.stack(im_batch)

        # collect random starting images for pgd
        im_n = []
        for _ in range(im_batch.shape[0]):
            idx = int(np.random.randint(0, len(train_loader.dataset)))
            im_n.append(train_loader.dataset.__getitem__(idx)[0])
        im_n = ch.stack(im_n)

        # get target feature vector
        with ch.no_grad():
            (_, rep), _ = model(im_batch.cuda(), with_latent=True)

        # perform pgd
        _, xadv = model(im_n.cuda(), rep.clone(), make_adv=True, **kwargs)

        # save as images
        for ii, path in enumerate(paths):
            save_path = os.path.join(args.save_dir, path.split('/')[-2])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, path.split('/')[-1])
            im = 255*xadv[ii].permute(1,2,0).detach().cpu().numpy()
            im = Image.fromarray(im.astype(np.uint8))
            im.save(save_path)

        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
