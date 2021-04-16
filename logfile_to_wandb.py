import os
import wandb
import argparse

'''
Script that goes through a given logfile and uploads to wandb

Before running this you need to create a project in wandb named 'edge-dropping'
And then `pip install wandb` and `wandb init`

'''

def upload_to_wandb(file_location, project_name, username, exp_name, dryrun=False):

    # Get Epoch, Train Acc, Test Acc, Train Loss, Test Loss
    assert os.path.exists(file_location), "Invalid path to log.txt" 
    with open(file_location, 'r') as f:
        lines = f.readlines()

    if not exp_name:
        exp_name = file_location.split('/')[-2]
    print('Exp name: ' + exp_name)

    if not dryrun:
        wandb.init(project=project_name, entity=username, name=exp_name)

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
           li.startswith("Argument") or \
           li.startswith("Epoch") or \
           li.startswith("Resuming"):
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


    if not dryrun:
        for i in range(len(epochs)):
            wandb.log({"Train Accuracy": train_acc[i],
                        "Test Accuracy": test_acc[i],
                        "Train Loss": train_loss[i],
                        "Cross-entropy Loss": xent_loss[i],
                        "Feature Loss": feat_loss[i]
                        }, step=epochs[i])
    else:
        print(f'Found: EPOCHS: {len(epochs)}, TRAIN ACCS {len(train_acc)}, TEST ACCS {len(test_acc)}, TRAIN LOSS: {len(train_loss)}, XENT LOSS: {len(xent_loss)}, FEAT LOSS: {len(feat_loss)}')


def main():
    parser = argparse.ArgumentParser(description="Parse log file and upload to wandb")
    parser.add_argument('--file-location', help='creates experiment with this name if not specified')
    parser.add_argument('--dryrun', action="store_true")
    parser.add_argument('--wandb-username', type=str, default='pratik18v')
    parser.add_argument('--wandb-project-name', type=str, default='robust-features')
    parser.add_argument('--wandb-exp-name', type=str, default='')

    args = parser.parse_args()

    upload_to_wandb(args.file_location, args.wandb_project_name, args.wandb_username,
                    args.wandb_exp_name, args.dryrun)

#main()
