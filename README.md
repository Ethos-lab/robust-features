# robust-features

## Getting Started

1. Use requirements.txt or environment.yml to recreate virtual environment.
2. Example command to train a student-teacher model:

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset CIFAR \
    --dataroot /path/to/cifar \
    --batch-size 128 \
    --xent-weight 0.001 \
    --l2-weight 1.0 \
    --teacher-load-path /path/to/teacher/model.pt \
    --load-std-data \
    --student-arch resnet50 \
    --epochs 100 \
    --milestones 65,90
```

3. Example command to adversarially train teacher model using robustness toolbox:

```
CUDA_VISIBLE_DEVICES=0 python -m robustness.main \
    --out-dir checkpoints \
    --exp-name cifar_resnet50_l2_1_0 \
    --epochs 150 \
    --lr 0.1 \
    --step-lr 50 \
    --adv-train 1 \
    --log-iters 10 \
    --save-ckpt-iters 10 \
    --dataset cifar \
    --data /path/to/cifar \
    --arch resnet50 \
    --batch-size 64 \
    --workers 4 \
    --data-aug 1 \
    --attack-steps 7 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

4. Example command to test a model that was trained using `train.py`:

```
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset CIFAR \
    --dataroot /path/to/cifar \
    --batch-size 50 \
    --model resnet50 \
    --load-path /path/to/model.pt \
    --model-type nn_module
```

5. Example command to test a model trained using robustness toolbox:

```
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset CIFAR \
    --dataroot /data/cifar \
    --batch-size 50 \
    --model resnet50 \
    --load-path /path/to/model.pt \
    --model-type attacker_model
```
