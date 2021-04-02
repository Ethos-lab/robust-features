# robust-features

## Getting Started

1. Use requirements.txt or environment.yml to recreate virtual environment
2. Download clone robustness toolbox into main directory from: https://github.com/MadryLab/robustness/tree/79d371fd799885ea5fe5553c2b749f41de1a2c4e
3. Example command to train a model:

`CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR --dataroot /data/cifar --batch-size 64 --xent-weight 0.001 --l2-weight 1 --rob-model-path cifar_resnet50_l2_1_0 --load-std-data --model resnet152 --epochs 150 --milestones 50,100 --normalize-l2-loss --exp-name l2_loss_normalized`

4. Example command to test a model that was trained using `train.py`:

`CUDA_VISIBLE_DEVICES=0 python test.py --dataset CIFAR --dataroot /data/cifar --batch-size 50 --model resnet50 --load-path /path/to/model.pt --model-type nn_module`

5. Example command to test a model provided by authors:

`CUDA_VISIBLE_DEVICES=0 python test.py --dataset CIFAR --dataroot /data/cifar --batch-size 50 --model resnet50 --load-path /path/to/model.pt --model-type attacker_model`
