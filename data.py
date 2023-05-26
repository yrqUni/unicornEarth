import numpy as np

import torch
from torch.utils.data import Dataset

class ERA5(Dataset):
    def __init__(self,data,num_patches,pretrain,target_num_patches,pad_mask,per_var_patch_side,pretrain_mask_rate):
        self.data = data
        self.num_patches = num_patches
        self.pretrain = pretrain    
        self.target_num_patches = target_num_patches
        self.pad_mask = pad_mask  
        self.per_var_patch_side = per_var_patch_side 
        self.pretrain_mask_rate = pretrain_mask_rate 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx][np.newaxis,:,:]
        if self.pretrain==True:
            bool_masked_pos = torch.zeros(size=(self.num_patches,))
            bool_masked_pos[:int(self.num_patches*self.pretrain_mask_rate)] = 1
            idx = torch.randperm(bool_masked_pos.nelement())
            bool_masked_pos_origin_shape = bool_masked_pos.size()
            bool_masked_pos = bool_masked_pos.view(-1)[idx].view(bool_masked_pos_origin_shape).bool() # bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
        if self.pretrain==False:
            target_num_patches_side = int(self.target_num_patches**0.5)
            num_patches_side = int(self.num_patches**0.5)
            bool_masked_pos = np.zeros((num_patches_side,num_patches_side))
            bool_masked_pos[:target_num_patches_side,:target_num_patches_side] = 1 
            bool_masked_pos = bool_masked_pos.flatten()
            bool_masked_pos = bool_masked_pos.astype(bool) # bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
        # pad_mask = torch.tensor(self.pad_mask).repeat_interleave(self.per_var_patch_side,0).repeat_interleave(self.per_var_patch_side,1).contiguous().flatten().bool()
        # a = torch.arange(0,36,1) a = a.reshape(6,6) a.repeat_interleave(8,0).repeat_interleave(8,1).contiguous().flatten()[384] tensor(6)
        return {'sample':sample[:,0,:,:], # (N, 1, 768, 768)
                'GT':sample[:,1,:,:], 
                'mask':bool_masked_pos,}
                # 'pad_mask':pad_mask}