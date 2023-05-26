
import os 
import argparse
import joblib
import math
import numpy as np
import time

from sklearn.model_selection import train_test_split 

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import SchedulerType, get_scheduler

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from utils.utils import print_rank_0, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_hf_format, save_zero_three_model, just_show
from utils.ds_utils import get_train_ds_config
from data import ERA5
from model import create_Init_model, create_from_PT_model
from lossFun import mask_l1_loss

def parse_args():
    parser = argparse.ArgumentParser(description="unicornEarth")
    # input 
    parser.add_argument('--data_sample_input_path', type=str, default='/public/home/hydeng/Workspace/yrqUni/unicornEarth/DATA/Merge/', help='')
    parser.add_argument('--data_padmask_input_path', type=str, default='/public/home/hydeng/Workspace/yrqUni/unicornEarth/DATA/PadMask/', help='')
    parser.add_argument('--val_rate', type=float, default=None, help='')
    parser.add_argument('--data_info', type=str, default='/public/home/hydeng/Workspace/yrqUni/unicornEarth/data/DataInfo', help='')
    parser.add_argument("--target_num_patches",type=int,default=64,help='')
    parser.add_argument("--per_var_patch_side",type=int,default=8,help='var side / patch side')
    # model init
    parser.add_argument("--init_model",type=str,default='unicornEarth',help='')
    parser.add_argument("--pretrain_model",type=str,default=None,help='')
    # train conf
    parser.add_argument("--per_device_train_batch_size",type=int,default=2,help='',)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=2,help='',)
    parser.add_argument("--pretrain",action='store_true',help='')
    parser.add_argument("--do_eval",action='store_true',help='')
    parser.add_argument("--pretrain_mask_rate", type=float, default=None, help='')
    # train learn conf
    parser.add_argument("--weight_decay",type=float,default=0.,help='')
    parser.add_argument("--num_train_epochs",type=int,default=1000,help='')
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help='')
    parser.add_argument('--gradient_checkpointing',action='store_true',help='')
    parser.add_argument("--learning_rate",type=float,default=1e-3,help='',)
    parser.add_argument("--lr_scheduler_type",type=SchedulerType,default="cosine",help="The scheduler type to use.",choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"],)
    parser.add_argument("--num_warmup_steps",type=int,default=0,help='')
    # output
    parser.add_argument("--ckpt_output_dir",type=str,default='./ckpt',help='')
    parser.add_argument('--data_output_path', type=str,default='',help='')
    # random
    parser.add_argument("--seed",type=int,default=1234,help='')
    # model config
    parser.add_argument('--disable_dropout',action='store_true',help='')
    # precision
    parser.add_argument('--use_fp16',action='store_true',help='')
    # parallel
    parser.add_argument("--local_rank",type=int,default=-1,help='')
    # ZeRO
    parser.add_argument('--offload', action='store_true', help='')
    parser.add_argument('--zero_stage', type=int, default=0, help='')
    # log
    parser.add_argument("--log_step", type=int, default=1, help='')
    parser.add_argument("--save_step", type=int, default=10, help='')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['fp16']["enabled"] = args.use_fp16

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    if args.pretrain_model!=None:
        model = create_from_PT_model(args.pretrain_model, disable_dropout=args.disable_dropout)
    if args.pretrain_model==None:
        model = create_Init_model(args.init_model, disable_dropout=args.disable_dropout)
    num_patches = (model.config.image_size // model.config.patch_size) ** 2
    patch_size = model.config.patch_size

    mask_l1_loss_fn = mask_l1_loss(model.config.patch_size, model.config.image_size, model.config.num_channels)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    len_data = 0
    data_info = joblib.load(args.data_info)
    for infos_key in data_info['sample']:
        infos = data_info['sample'][infos_key]
        len_data = len_data+infos[0]
    len_train_dataloader = int(len_data*(1-args.val_rate))
    print_rank_0(f'All len train dataloader:{len_train_dataloader}',args.global_rank)

    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, 
                                 num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_train_epochs * num_update_steps_per_epoch, )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, args=args, 
                                                             config=ds_config, lr_scheduler=lr_scheduler, dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    start_log_time = time.time()
    for epoch in range(args.num_train_epochs):
        
        torch.distributed.barrier()
        print_rank_0(f'Use data in list {os.listdir(args.data_sample_input_path)}, len is {len(os.listdir(args.data_sample_input_path))}',args.global_rank)
        P = 0
        for data_part in os.listdir(args.data_sample_input_path):
            P = P+1
            data_path = f'{args.data_sample_input_path}/{data_part}'
            PadMask = joblib.load(f'{args.data_padmask_input_path}/{data_part}')
            print_rank_0(f'Use {data_path} now',args.global_rank)
            data = joblib.load(os.path.join(data_path))
            trainData, valData, _, _ = train_test_split(data,np.ones(data.shape[0]),test_size=args.val_rate, random_state=42)
            TrDataset = ERA5(trainData,num_patches,args.pretrain,args.target_num_patches,PadMask,args.per_var_patch_side,args.pretrain_mask_rate)
            ValDataset = ERA5(valData,num_patches,args.pretrain,args.target_num_patches,PadMask,args.per_var_patch_side,args.pretrain_mask_rate) 
            if args.local_rank == -1:
                train_sampler = RandomSampler(TrDataset)
                eval_sampler = SequentialSampler(ValDataset)
            else:
                train_sampler = DistributedSampler(TrDataset)
                eval_sampler = DistributedSampler(ValDataset)
            train_dataloader = DataLoader(TrDataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
            eval_dataloader = DataLoader(ValDataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)
            torch.distributed.barrier()

            def evaluation(model, eval_dataloader):
                model.eval()
                losses = 0
                for step, batch in enumerate(eval_dataloader):
                    sample = batch['sample'].to(device) # (N, 1, 768, 768)
                    GT = batch['GT'].to(device) # (N, 1, 768, 768)
                    mask = batch['mask'].to(device) # (N, num_patch)
                    # pad_mask = batch['pad_mask'].to(device) # (N, num_patch)
                    with torch.no_grad():
                        outputs = model(sample, bool_masked_pos=mask)
                    loss1, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                    loss2 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                    losses += loss2.float()
                losses = losses / (step + 1)
                losses = get_all_reduce_mean(losses).item()
                return losses, reconstructed_pixel_values, sample

            # Train!
            print_rank_0("***** Running training *****", args.global_rank)
            if args.do_eval:
                print_rank_0(f"***** Evaluating, Epoch {0}/{args.num_train_epochs} *****", args.global_rank)
                val_loss,_,_ = evaluation(model, eval_dataloader)
                print_rank_0(f"val loss: {val_loss}", args.global_rank)
            
            print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}", args.global_rank)
            training_step_losses = []
            model.train()
            for step, batch in enumerate(train_dataloader):
                sample = batch['sample'].to(device) # (N, 1, 768, 768)
                GT = batch['GT'].to(device) # (N, 1, 768, 768)
                mask = batch['mask'].to(device) # (N, num_patch)
                # pad_mask = batch['pad_mask'].to(device) # (N, num_patch)
                outputs = model(sample, bool_masked_pos=mask)
                loss1, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                loss2 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                model.backward(loss2)
                model.step()
                training_step_losses.append(loss2)
                if step%args.log_step == 0:
                    end_log_time = time.time()
                    log_time = end_log_time-start_log_time
                    _loss = sum(training_step_losses)/len(training_step_losses)
                    _log_step = (epoch*len(train_dataloader))+step+1
                    _speed = (log_time)/((epoch*len(train_dataloader))+step+1)
                    _train_schedule = ((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader))
                    _all_to_consume = (log_time)/(((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader)))
                    _estimated_to_consume = ((log_time)/(((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader))))*(1-(((epoch*len(train_dataloader))+step+1)/(args.num_train_epochs*len(train_dataloader))))
                    print_rank_0(f"epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} step {step} train loss {_loss}, log_step {_log_step}, speed {_speed}, train schedule {_train_schedule}, all to consume {_all_to_consume}, estimated to consume {_estimated_to_consume}", args.global_rank)
                    just_show(reconstructed_pixel_values,sample,patch_size,args.per_var_patch_side,args.data_output_path)
                    training_step_losses = []
                if step%args.save_step == 0 and args.global_rank == 0 and args.ckpt_output_dir is not None:
                    save_hf_format(model, args)
            # Evaluate perplexity on the validation set.
            if args.do_eval:
                print_rank_0(f"***** Evaluating, Epoch {0}/{args.num_train_epochs} *****", args.global_rank)
                val_loss,_,_ = evaluation(model, eval_dataloader)
                print_rank_0(f"val loss: {val_loss}", args.global_rank)                
            model.tput_timer.update_epoch_count()

    if args.ckpt_output_dir is not None:
        os.makedirs(os.path.abspath(os.path.dirname(args.ckpt_output_dir)), exist_ok=True)
        print_rank_0('saving the final model ...', args.global_rank)
        if args.global_rank == 0:
            save_hf_format(model, args)
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model, args.global_rank, args.ckpt_output_dir, zero_stage=args.zero_stage)
    
    torch.distributed.barrier()
    print_rank_0('ALL DONE!!!', args.global_rank)

if __name__ == "__main__":
    main()
    print('=== exit normally ===')
