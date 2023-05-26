from torch import nn

class mask_l1_loss():
    def __init__(self,patch_size,image_size,num_channels):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
    def compute(self,pixel_values=None,reconstructed_pixel_values=None,bool_masked_pos=None): 
        size = self.image_size // self.patch_size
        bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
        mask = (
            bool_masked_pos.repeat_interleave(self.patch_size, 1)
            .repeat_interleave(self.patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )
        reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
        masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.num_channels
        return masked_im_loss
