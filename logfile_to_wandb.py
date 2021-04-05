import os
import wandb
import argparse

'''
Script that goes through a given logfile and uploads to wandb

Before running this you need to create a project in wandb named 'edge-dropping'
And then `pip install wandb` and `wandb init`

Pretty brute force, expects normal log file (no breaks/restarts or any other prints)

'''

parser = argparse.ArgumentParser(description="Parse log file and upload to wandb")
parser.add_argument('--file-location', help='creates experiment with this name if not specified')
parser.add_argument('--wandb-exp-name', type=str, default='')
parser.add_argument('--dryrun', action="store_true")

args = parser.parse_args()


if not args.wandb_exp_name:
    args.wandb_exp_name = args.file_location.split('/')[-2]
print('Exp name: ' + args.wandb_exp_name)


if not args.dryrun:
    wandb.init(project="robust-features", entity="pratik18v", name=args.wandb_exp_name)

# Get Epoch, Train Acc, Test Acc, Train Loss, Test Loss
with open(args.file_location, 'r') as f:
    lines = f.readlines()

epochs, train_acc, test_acc, train_loss = [], [], [], []
xent_loss, feat_loss = [], []

ctr = 0
for l in lines:
    li = l.strip()

    if not li:
        break 

    if li == "Started Training !!!" or \
       li == "Saving model !!!" or \
       li == "Finished Training !!!" or \
       li.startswith("Argument"):
        continue

    tokens = li.split(' ')
    if tokens[0] == "Train":
        train_loss.append(float(tokens[3]))
        xent_loss.append(float(tokens[4].replace("(", "").replace(",", "")))
        feat_loss.append(float(tokens[5].replace(")", "").replace(",", "")))
        train_acc.append(float(tokens[-1].replace("%", "")))
        epochs.append(ctr)
        ctr += 1

    elif tokens[0] == "Val":
        test_acc.append(float(tokens[4].replace("%", "").replace(",", "")))


if not args.dryrun:
    for i in range(len(epochs)):
        wandb.log({"Train Accuracy": train_acc[i],
                    "Test Accuracy": test_acc[i],
                    "Train Loss": train_loss[i],
                    "Cross-entropy Loss": xent_loss[i],
                    "Feature Loss": feat_loss[i]
                    }, step=epochs[i])
else:
    print(f'Found: EPOCHS: {len(epochs)}, TRAIN ACCS {len(train_acc)}, TEST ACCS {len(test_acc)}, TRAIN LOSS: {len(train_loss)}, XENT LOSS: {len(xent_loss)}, FEAT LOSS: {len(feat_loss)}')
