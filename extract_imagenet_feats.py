import os
import tqdm
import argparse
from PIL import Image

import torch as ch
from torchvision import transforms

from robustness import imagenet_models
import robustness.data_augmentation as da
from utils import load_checkpoint, ModelwithInputNormalization

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def get_args():
    parser = argparse.ArgumentParser(description="Robust data generation script.")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/imagenet/train",
                        help="path to dataset")
    parser.add_argument("--model-dir", type=str, default="rimagenet_alexnet_l2_3_0")

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
    args.load_path = f"./checkpoints/{args.model_dir}/checkpoint.pt.best"

    args.save_dir = f"{args.dataroot.replace('train', 'train_feats')}/{args.model_dir}"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    model = imagenet_models.__dict__[args.model](num_classes=9)

    ckpt = load_checkpoint(args.load_path)
    model.load_state_dict(ckpt['model_sd'])
    print(f"Successfully loaded checkpoint from epoch: {ckpt['load_epoch']} !!!")

    mean = ch.tensor([0.4717, 0.4499, 0.3837])
    std = ch.tensor([0.2600, 0.2516, 0.2575])
    model = ModelwithInputNormalization(model, mean, std).cuda()
    model.eval()

    class_keys = os.listdir(args.dataroot)
    pbar = tqdm.tqdm(total=len(class_keys))
    for target in class_keys:
        d = os.path.join(args.dataroot, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for ii, fname in enumerate(sorted(fnames)):
                pbar.set_description(f"Progress: {ii+1}/{len(fnames)}")
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    with open(path, 'rb') as f:
                        img = Image.open(f)
                        img = img.convert('RGB')
                    img = da.TEST_TRANSFORMS_IMAGENET(img)
                    img = img.unsqueeze(0)

                    with ch.no_grad():
                        _, rep = model(img.cuda(), with_latent=True)

                    save_path = os.path.join(args.save_dir, fname.split('_')[0])
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    save_path = os.path.join(save_path, fname.replace('.JPEG', '.feat'))
                    ch.save(rep.squeeze(0).cpu(), save_path)

        pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    main()
