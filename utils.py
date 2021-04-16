import dill
import torch as ch

def load_checkpoint(load_path, mode='eval'):
    assert mode in ['train', 'eval']

    checkpoint = ch.load(load_path, pickle_module=dill)
    state_dict_path = "model"
    if not ("model" in checkpoint):
        state_dict_path = "state_dict"

    sd = checkpoint[state_dict_path]
    sd = {k.replace("module.model.", ""):v for k,v in sd.items()\
          if ("attacker" not in k) and ("normalizer" not in k)}

    load_epoch = checkpoint['epoch']
    optim_sd = None
    best_acc = None
    r_time_avg = None
    if mode == 'train':
        optim_sd = checkpoint["optimizer_state_dict"]
        best_acc = checkpoint["best_acc"]
        r_time_avg = checkpoint["time_per_epoch"]

    # clear out memory
    del checkpoint

    ckpt = {'model_sd': sd,
            'load_epoch': load_epoch,
            'optim_sd': optim_sd,
            'best_acc': best_acc,
            'r_time_avg': r_time_avg}

    return ckpt 
