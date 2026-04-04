import sys
import torch
from torch import nn as nn
from basicsr.utils.registry import LOSS_REGISTRY

sys.path.append("..")        
from varmodel import VQVAE

@LOSS_REGISTRY.register()
class PriorLoss1(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(PriorLoss1, self).__init__()
    
        self.loss_weight = loss_weight
        
    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        
        #loss_abs = 0
        
        loss_abs = nn.L1Loss()(S2_fea, S1_fea.detach())
        return self.loss_weight * loss_abs

@LOSS_REGISTRY.register()
class PriorLoss2(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(PriorLoss2, self).__init__()
    
        self.loss_weight = loss_weight
        
    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        
        #loss_abs = 0
        
        loss_abs = nn.L1Loss()(S2_fea, S1_fea.detach())
        return self.loss_weight * loss_abs
    
@LOSS_REGISTRY.register()
class MSVAELoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight1=1.0, loss_weight2=1.0,vae_ckpt='your-path/vae_ch160v4096z32.pth'):
        super(MSVAELoss, self).__init__()
        patch_nums = (1, 4, 8, 16)
        vae_ckpt= vae_ckpt
        self.vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        self.patch_nums = patch_nums
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        self.loss_weight1 = loss_weight1
        self.loss_weight2 = loss_weight2
        for param in self.vae.parameters():
            param.requires_grad = False

    def normalize_01_into_pm1(self, x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    
    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        
        #loss_abs = 0
        S1_fea_img_normal = self.normalize_01_into_pm1(S1_fea)
        S1_fea_img_tokens,encoder_S1_fea = self.vae.img_to_idxBl(S1_fea_img_normal, self.patch_nums)
        S1_fea_hat = []
        for index, img_token in enumerate(S1_fea_img_tokens):
            S1_fea_tokens_embedding = self.vae.quantize.embedding(img_token)
            fft_p_i = torch.fft.fft2(S1_fea_tokens_embedding)
            #print("fft_p_i.shape", fft_p_i.shape)
            S1_fea_hat.append(fft_p_i.real)
            S1_fea_hat.append(fft_p_i.imag)
        S1_fea_fft = torch.cat(S1_fea_hat,dim=1)

        S2_fea_hat = []
        S2_fea_img_normal = self.normalize_01_into_pm1(S2_fea)
        S2_fea_img_tokens,encoder_S2_fea = self.vae.img_to_idxBl(S2_fea_img_normal, self.patch_nums)
        S2_fea_hat = []
        for index, img_token in enumerate(S2_fea_img_tokens):
            S2_fea_tokens_embedding = self.vae.quantize.embedding(img_token)
            fft_p_i = torch.fft.fft2(S2_fea_tokens_embedding)
            S2_fea_hat.append(fft_p_i.real)
            S2_fea_hat.append(fft_p_i.imag)
        S2_fea_fft = torch.cat(S2_fea_hat,dim=1)
        loss_abs_ms = nn.L1Loss()(S2_fea_fft, S1_fea_fft)
        loss_abs_fea = nn.L1Loss()(encoder_S2_fea, encoder_S1_fea)
        return self.loss_weight1 * loss_abs_ms, self.loss_weight2 * loss_abs_fea

