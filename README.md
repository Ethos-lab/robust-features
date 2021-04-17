# robust-features

## Getting Started

1. Use requirements.txt or environment.yml to recreate virtual environment.

2. Adversarially train teacher model using robustness toolbox (cifar10 and restricted_imagenet):

```
CUDA_VISIBLE_DEVICES=0 python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --out-dir checkpoints \
    --exp-name cifar_resnet50_l2_1_0 \
    --arch resnet50 \
    --adv-train 1 \
    --adv-eval 1 \
    --epochs 150 \
    --lr 0.1 \
    --step-lr 50 \
    --batch-size 128 \
    --workers 4 \
    --data-aug 1 \    
    --log-iters 1 \
    --save-ckpt-iters 1 \
    --attack-steps 7 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```
The best checkpoint will be saved at: `{out_dir}/{exp_name}/checkpoint.pt.best`.
NOTE: Set attack_lr to eps/5.

3. Extract robust features (restricted_imagenet only):
```
CUDA_VISIBLE_DEVICES=0 python extract_imagenet_feats.py \
    --dataroot /path/to/imagenet/train \
    --model-dir rimagenent_alexnet_l2_3_0
```

4. (a) Train a student using the robust teacher model (cifar10):

```
CUDA_VISIBLE_DEVICES=0 python train_cifar.py \
    --dataroot /path/to/cifar \
    --load-std-data \
    --teacher-load-path /path/to/teacher/model.pt \
    --student-arch resnet50 \
    --xent-weight 0.001 \
    --l2-weight 1.0 \
    --epochs 100 \
    --milestones 65,90 \
    --lr 0.1 \
    --batch-size 128
```
4. (b) Train a student using the robust teacher model (imagenet):

```
CUDA_VISIBLE_DEVICES=0 python train_imagenet.py \
    --dataroot /path/to/imagenet \
    --load-std-data \
    --load-feats rimagenet_alexnet_l2_3_0 \
    --student-arch resnet50 \
    --xent-weight 0.001 \
    --l2-weight 1.0 \
    --epochs 100 \
    --milestones 90 \
    --lr 0.01 \
    --batch-size 64
```
The best checkpoint will be saved at: `./checkpoints/{dataset}-feat_loss=l2-teacher={teacher_arch}-student={student_arch}-xent_weight={xent_weight}-l2_weight={l2_weight}-{exp_name (optional)}/checkpoint.pt.best`.

If the value for arg `save_dir` is provided, best checkpoint will be saved at: `{save_dir}/checkpoint.pt.best`.

5. Example command to test a model:
```
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset cifar \
    --dataroot /path/to/cifar \
    --batch-size 50 \
    --model resnet50 \
    --load-path /path/to/model.pt \
    --eps 0.5 \
    --step-size 0.03125 \
    --pgd-iters 20 \
    --constraint 2
```

6. Generate robust dataset (D_r) (both cifar and restricted_imagenet):
```
CUDA_VISIBLE_DEVICES=0 python generate_dr.py \
    --dataset cifar \
    --dataroot /path/to/cifar \
    --batch-size 50 \
    --model-dir cifar_resnet50_l2_1_0 \
    --step-size 0.1 \
    --iterations 1000
```
Robust dataset will be saved at `{dataroot}/(model_dir}`

7. To train with robust data, use same commands as 4. Replace arg `load_std_data` with arg `rob_dataroot`. Pass save path from previous step to this arg.
