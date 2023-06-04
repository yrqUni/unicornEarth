
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
from model import create_Init_ViT_model, create_from_PT_ViT_model, create_Init_SwinTransV2_model, create_from_PT_SwinTransV2_model
from lossFun import mask_l1_loss, ssim, ms_ssim, SSIM, MS_SSIM

def parse_args():
    parser = argparse.ArgumentParser(description="unicornEarth")
    # input 
    parser.add_argument('--data_sample_input_path', type=str, default='../DATA/Merge/', help='')
    parser.add_argument('--data_padmask_input_path', type=str, default='../DATA/PadMask/', help='')
    parser.add_argument('--val_rate', type=float, default=None, help='')
    parser.add_argument('--data_info', type=str, default='./data/DataInfo', help='')
    parser.add_argument("--target_num_patches",type=int,default=64,help='')
    parser.add_argument("--patch_per_var_side",type=int,default=8,help='var side / patch side')
    parser.add_argument("--stats_path",type=str,default='./data/Stats/',help='')
    parser.add_argument("--target_var",type=str,default='TCWV',help='')
    # model init
    parser.add_argument("--model",type=str,default=None,help='ViT SwinV1')
    parser.add_argument("--init_model",type=str,default='unicornEarth',help='')
    parser.add_argument("--pretrain_model",type=str,default=None,help='')
    # train conf
    parser.add_argument("--per_device_train_batch_size",type=int,default=2,help='',)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=2,help='',)
    parser.add_argument("--train_stage", type=str, default=None, help='')
    parser.add_argument("--do_eval",action='store_true',help='')
    parser.add_argument("--pretrain_mask_rate", type=float, default=None, help='')
    parser.add_argument("--loss_l1_rate", type=float, default=None, help='')
    parser.add_argument("--loss_ms_ssim_rate", type=float, default=None, help='')
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
    if args.model=='ViT':
        if args.pretrain_model!=None:
            model = create_from_PT_ViT_model(args.pretrain_model, disable_dropout=args.disable_dropout)
        if args.pretrain_model==None:
            model = create_Init_ViT_model(args.init_model, disable_dropout=args.disable_dropout)
    if args.model=='SwinV1':
        if args.pretrain_model!=None:
            model = create_from_PT_SwinTransV2_model(args.pretrain_model, disable_dropout=args.disable_dropout)
        if args.pretrain_model==None:
            model = create_Init_SwinTransV2_model(args.init_model, disable_dropout=args.disable_dropout)
    num_patches = (model.config.image_size // model.config.patch_size) ** 2
    image_size = model.config.image_size 
    patch_size = model.config.patch_size
    
    len_data = 0
    data_info = joblib.load(args.data_info)
    for infos_key in data_info['sample']:
        infos = data_info['sample'][infos_key]
        len_data = len_data+infos[0]
    len_train_data = int(len_data*(1-args.val_rate))
    len_train_dataloader = len_train_data // (args.per_device_train_batch_size * torch.distributed.get_world_size())
    print_rank_0(f'All len train dataloader:{len_train_dataloader}',args.global_rank)

    target_data_stats = joblib.load(f'{args.stats_path}/{args.target_var}')
    print_rank_0(f'target var is {args.target_var}, Min is {target_data_stats["Min"]}, Max is {target_data_stats["Max"]}',args.global_rank)

    mask_l1_loss_fn = mask_l1_loss(model.config.patch_size, model.config.image_size, model.config.num_channels)
    ms_ssim_loss_fn = MS_SSIM(data_range=target_data_stats["Max"], size_average=True, channel=1)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, 
                                 num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_train_epochs * num_update_steps_per_epoch, )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, args=args, 
                                                             config=ds_config, lr_scheduler=lr_scheduler, dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    start_log_time = time.time()
    for epoch in range(args.num_train_epochs):
        stepInEp = 0
        torch.distributed.barrier()
        print_rank_0(f'################ Use data in list {os.listdir(args.data_sample_input_path)}, len is {len(os.listdir(args.data_sample_input_path))} ################',args.global_rank)
        P = 0
        for data_part in os.listdir(args.data_sample_input_path):
            P = P+1
            data_path = f'{args.data_sample_input_path}/{data_part}'
            PadMask = joblib.load(f'{args.data_padmask_input_path}/{data_part}')
            print_rank_0(f'################ Use {data_path} now ################',args.global_rank)
            data = joblib.load(os.path.join(data_path))
            trainData, valData, _, _ = train_test_split(data,np.ones(data.shape[0]),test_size=args.val_rate, random_state=args.seed, shuffle=False)
            TrDataset = ERA5(trainData,num_patches,args.train_stage,args.target_num_patches,PadMask,args.patch_per_var_side,args.pretrain_mask_rate)
            ValDataset = ERA5(valData,num_patches,args.train_stage,args.target_num_patches,PadMask,args.patch_per_var_side,args.pretrain_mask_rate) 
            if args.local_rank == -1:
                train_sampler = RandomSampler(TrDataset)
                eval_sampler = SequentialSampler(ValDataset)
            else:
                train_sampler = DistributedSampler(TrDataset)
                eval_sampler = DistributedSampler(ValDataset)
            train_dataloader = DataLoader(TrDataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
            eval_dataloader = DataLoader(ValDataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)
            torch.distributed.barrier()

            def evaluation(args, model, eval_dataloader):
                model.eval()
                losses_l1 = 0
                losses_ms_ssim = 0
                losses_mix = 0
                for step, batch in enumerate(eval_dataloader):
                    sample = batch['sample'].float().to(device) # (N, 1, 768, 768)
                    GT = batch['GT'].float().to(device) # (N, 1, 768, 768)
                    mask = batch['mask'].to(device) # (N, num_patch)
                    size = image_size//patch_size
                    mask_expand = mask.reshape(-1, size, size)
                    mask_expand = (mask_expand.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1).contiguous()).float().to(device)
                    # pad_mask = batch['pad_mask'].to(device) # (N, num_patch)
                    with torch.no_grad():
                        if args.train_stage=='FT':
                            none_mask = batch['none_mask'].to(device) # (N, num_patch)
                            outputs = model(sample, bool_masked_pos=none_mask)
                            _, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                            loss_l1 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                            loss_ms_ssim = 1-ms_ssim_loss_fn((reconstructed_pixel_values*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side],(GT*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side])
                            loss_mix = args.loss_l1_rate*loss_l1+args.loss_ms_ssim_rate*loss_ms_ssim
                        if args.train_stage=='PT2':
                            outputs = model(sample, bool_masked_pos=mask)
                            _, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                            loss_l1 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                            loss_ms_ssim = 1-ms_ssim_loss_fn((reconstructed_pixel_values*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side],(GT*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side])
                            loss_mix = args.loss_l1_rate*loss_l1+args.loss_ms_ssim_rate*loss_ms_ssim
                        if args.train_stage=='PT1':
                            outputs = model(sample, bool_masked_pos=mask)
                            _, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                            loss_l1 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                    losses_l1 += loss_l1.float()
                    if args.train_stage=='PT2' or args.train_stage=='FT':
                        losses_ms_ssim += loss_ms_ssim.float()
                        losses_mix += loss_mix.float()
                losses_l1 = losses_l1 / (step + 1)
                losses_l1 = get_all_reduce_mean(losses_l1).item()
                if args.train_stage=='PT2' or args.train_stage=='FT':
                    losses_ms_ssim = losses_ms_ssim / (step + 1)
                    losses_ms_ssim = get_all_reduce_mean(losses_ms_ssim).item()
                    losses_mix = losses_mix / (step + 1)
                    losses_mix = get_all_reduce_mean(losses_mix).item()
                if args.global_rank==0:
                    just_show(reconstructed_pixel_values,sample,patch_size,args.patch_per_var_side,f'{args.data_output_path}/valVis/')
                if args.train_stage=='PT1':
                    return losses_l1, reconstructed_pixel_values, sample
                if args.train_stage=='PT2' or args.train_stage=='FT':
                    return losses_l1, losses_ms_ssim, losses_mix, reconstructed_pixel_values, sample

            # Train!
            if args.do_eval:
                if args.train_stage=='PT1':
                    losses_l1, _, _ = evaluation(args, model, eval_dataloader)
                    print_rank_0(f">>>>>>>>>>>>>>>> Beginning epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} val losses_l1: {losses_l1} <<<<<<<<<<<<<<<<", args.global_rank)
                if args.train_stage=='PT2' or args.train_stage=='FT':
                    losses_l1, losses_ms_ssim, losses_mix, _, _ = evaluation(args, model, eval_dataloader)
                    print_rank_0(f">>>>>>>>>>>>>>>> Beginning epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} val losses_l1 {losses_l1}, val losses_ms_ssim {losses_ms_ssim}, val losses_mix {losses_mix} <<<<<<<<<<<<<<<<", args.global_rank)
            print_rank_0(f"################ Epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))}, Total Micro Batches {len_train_dataloader} ################", args.global_rank)
            training_step_losses_l1 = []
            training_step_losses_ms_ssim = []
            training_step_losses_mix = []
            model.train()
            for step, batch in enumerate(train_dataloader):
                stepInEp = stepInEp+1
                sample = batch['sample'].float().to(device) # (N, 1, 768, 768)
                GT = batch['GT'].float().to(device) # (N, 1, 768, 768)
                mask = batch['mask'].to(device) # (N, num_patch)
                size = image_size//patch_size
                mask_expand = mask.reshape(-1, size, size)
                mask_expand = (mask_expand.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1).contiguous()).float().to(device)
                # pad_mask = batch['pad_mask'].to(device) # (N, num_patch)
                if args.train_stage=='FT':
                    none_mask = batch['none_mask'].to(device) # (N, num_patch)
                    outputs = model(sample, bool_masked_pos=none_mask)
                    _, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                    loss_l1 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                    loss_ms_ssim = 1-ms_ssim_loss_fn((reconstructed_pixel_values*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side],(GT*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side])
                    loss_mix = args.loss_l1_rate*loss_l1+args.loss_ms_ssim_rate*loss_ms_ssim
                    model.backward(loss_mix)
                    model.step()
                    training_step_losses_l1.append(loss_l1)
                    training_step_losses_ms_ssim.append(loss_ms_ssim)
                    training_step_losses_mix.append(loss_mix)
                if args.train_stage=='PT2':
                    outputs = model(sample, bool_masked_pos=mask)
                    _, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                    loss_l1 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                    loss_ms_ssim = 1-ms_ssim_loss_fn((reconstructed_pixel_values*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side],(GT*mask_expand)[:,:,:patch_size*args.patch_per_var_side,:patch_size*args.patch_per_var_side])
                    loss_mix = args.loss_l1_rate*loss_l1+args.loss_ms_ssim_rate*loss_ms_ssim
                    model.backward(loss_mix)
                    model.step()
                    training_step_losses_l1.append(loss_l1)
                    training_step_losses_ms_ssim.append(loss_ms_ssim)
                    training_step_losses_mix.append(loss_mix)
                if args.train_stage=='PT1':
                    outputs = model(sample, bool_masked_pos=mask)
                    _, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
                    loss_l1 = mask_l1_loss_fn.compute(pixel_values=GT,reconstructed_pixel_values=reconstructed_pixel_values,bool_masked_pos=mask)
                    model.backward(loss_l1)
                    model.step()
                    training_step_losses_l1.append(loss_l1)
                if stepInEp%args.log_step == 0:
                    end_log_time = time.time()
                    log_time = end_log_time-start_log_time
                    _log_step = (epoch*len_train_dataloader)+stepInEp
                    _speed = (log_time)/((epoch*len_train_dataloader)+stepInEp)
                    _train_schedule = ((epoch*len_train_dataloader)+stepInEp)/(args.num_train_epochs*len_train_dataloader)
                    _all_to_consume = (log_time)/(((epoch*len_train_dataloader)+stepInEp)/(args.num_train_epochs*len_train_dataloader))
                    _estimated_to_consume = ((log_time)/(((epoch*len_train_dataloader)+stepInEp)/(args.num_train_epochs*len_train_dataloader)))*(1-(((epoch*len_train_dataloader)+stepInEp)/(args.num_train_epochs*len_train_dataloader)))
                    _log_step = get_all_reduce_mean(torch.tensor(_log_step).to(device)).item()
                    _speed = get_all_reduce_mean(torch.tensor(_speed).to(device)).item()
                    _train_schedule = get_all_reduce_mean(torch.tensor(_train_schedule).to(device)).item()
                    _all_to_consume = get_all_reduce_mean(torch.tensor(_all_to_consume).to(device)).item()
                    _estimated_to_consume = get_all_reduce_mean(torch.tensor(_estimated_to_consume).to(device)).item()
                    if args.train_stage=='PT1':
                        _loss_l1 = sum(training_step_losses_l1)/len(training_step_losses_l1)
                        _loss_l1 = get_all_reduce_mean(_loss_l1).item()
                        print_rank_0(f"epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} stepInEp {stepInEp} train l1_loss {_loss_l1}, log step {_log_step}, speed {_speed}, train schedule {_train_schedule}, all to consume {_all_to_consume}, estimated to consume {_estimated_to_consume}", args.global_rank)
                    if args.train_stage=='FT' or args.train_stage=='PT2':
                        _loss_l1 = sum(training_step_losses_l1)/len(training_step_losses_l1)
                        _loss_ms_ssim = sum(training_step_losses_ms_ssim)/len(training_step_losses_ms_ssim)
                        _loss_mix = sum(training_step_losses_mix)/len(training_step_losses_mix)
                        _loss_l1 = get_all_reduce_mean(_loss_l1).item()
                        _loss_ms_ssim = get_all_reduce_mean(_loss_ms_ssim).item()
                        _loss_mix = get_all_reduce_mean(_loss_mix).item()
                        print_rank_0(f"epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} stepInEp {stepInEp} train l1_loss {_loss_l1}, train mix_loss {_loss_mix}({args.loss_l1_rate}*loss_l1+{args.loss_ms_ssim_rate}*loss_ms_ssim), train sm_ssim_loss {_loss_ms_ssim}, log step {_log_step}, speed {_speed}, train schedule {_train_schedule}, all to consume {_all_to_consume}, estimated to consume {_estimated_to_consume}", args.global_rank)
                    if args.global_rank==0:
                        just_show(reconstructed_pixel_values,sample,patch_size,args.patch_per_var_side,f'{args.data_output_path}/trainVis/')
                    training_step_losses_l1 = []
                    training_step_losses_ms_ssim = []
                if stepInEp%args.save_step == 0 and args.global_rank == 0 and args.ckpt_output_dir is not None:
                    save_hf_format(model, args)
            # Evaluate perplexity on the validation set.
            if args.do_eval:
                if args.train_stage=='PT1':
                    losses_l1, _, _ = evaluation(args, model, eval_dataloader)
                    print_rank_0(f"<<<<<<<<<<<<<<<< End epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} val losses_l1: {losses_l1} >>>>>>>>>>>>>>>>", args.global_rank)
                if args.train_stage=='FT' or args.train_stage=='PT2':
                    losses_l1, losses_ms_ssim, losses_mix, _, _ = evaluation(args, model, eval_dataloader)
                    print_rank_0(f"<<<<<<<<<<<<<<<< End epoch {epoch} part {P}/{len(os.listdir(args.data_sample_input_path))} val losses_l1 {losses_l1}, val losses_ms_ssim {losses_ms_ssim}, val losses_mix {losses_mix} >>>>>>>>>>>>>>>>", args.global_rank)
        model.tput_timer.update_epoch_count()

    if args.ckpt_output_dir is not None:
        os.makedirs(os.path.abspath(os.path.dirname(args.ckpt_output_dir)), exist_ok=True)
        print_rank_0('saving the final model ...', args.global_rank)
        if args.global_rank == 0:
            save_hf_format(model, args)
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model, args.global_rank, args.ckpt_output_dir, zero_stage=args.zero_stage)
        print_rank_0('saving the final model DONE !!!', args.global_rank)
    
    torch.distributed.barrier()
    print_rank_0('ALL DONE !!!', args.global_rank)

if __name__ == "__main__":
    main()
    print('=== exit normally ===')
