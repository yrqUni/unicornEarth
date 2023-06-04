import os
import gc
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import set_seed
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters

def save_hf_format(model, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    ckpt_output_dir = os.path.join(args.ckpt_output_dir, sub_folder)
    if not os.path.exists(ckpt_output_dir):
        os.makedirs(ckpt_output_dir)
    output_model_file = os.path.join(ckpt_output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(ckpt_output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)


def just_show(reconstructed_pixel_values,sample,patch_size,per_var_patch_side,output_path):
    os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
    luck1 = random.randint(0,sample.shape[0]-1)
    luck2 = random.randint(0,sample.shape[1]-1)
    plt.matshow(np.squeeze(reconstructed_pixel_values[luck1,luck2,:patch_size*per_var_patch_side,:patch_size*per_var_patch_side].cpu().detach().numpy()))
    plt.colorbar()
    plt.savefig(f'{output_path}/reconstructed_pixel_values.jpg')
    plt.close()
    plt.cla()
    plt.clf()
    gc.collect()
    plt.matshow(np.squeeze(sample[luck1,luck2,:patch_size*per_var_patch_side,:patch_size*per_var_patch_side].cpu().detach().numpy()))
    plt.colorbar()
    plt.savefig(f'{output_path}/sample.jpg')
    plt.close()
    plt.cla()
    plt.clf()
    gc.collect()
    
