import MSRestoreX.archs.common as common
from ldm.ddpm_or import DDPM
from ldm.ddpm_or2 import DDPM2
#import MSRestoreX.archs.attention as attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

import sys
import math
sys.path.append("..")        
from varmodel import VQVAE
from thop import profile



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
        
    def forward(self, x,p_i):
        
        b,c,h,w = x.shape
        ''''''
        #p_i = self.E(p_i).view(b,-1)
        p_i=self.kernel(p_i).view(-1,c*2,1,1)
        p_i1,p_i2=p_i.chunk(2, dim=1)
        x = x*p_i1+p_i2  
        
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MSFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(MSFeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*4, kernel_size=1, bias=bias)
        
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0,
                      groups=hidden_features),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                      groups=hidden_features),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=2,
                      groups=hidden_features),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=7, padding=3,
                      groups=hidden_features),
        )

        self.project_out = nn.Conv2d(hidden_features*4, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(1024, dim*2, bias=False),
        )
        
    def forward(self, x,p_i):
        b,c,h,w = x.shape
        p_i=self.kernel(p_i).view(-1,c*2,1,1)
        p_i1,p_i2=p_i.chunk(2, dim=1)
        x = x*p_i1+p_i2  
        x = self.project_in(x)
        x_0,x_1,x_2,x_3 = x.chunk(4, dim=1)
        x_0 = self.conv1_0(x_0)
        x_1 = self.conv1_1(x_1)
        x_2 = self.conv1_2(x_2)
        x_3 = self.conv1_3(x_3)
        x = torch.cat([x_0,x_1,x_2,x_3],dim=1)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
        
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x,p_i):
        
        b,c,h,w = x.shape
        
        p_i=self.kernel(p_i).view(-1,c*2,1,1)
        p_i1,p_i2=p_i.chunk(2, dim=1)
        x = x*p_i1+p_i2  
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class FFTAttention(nn.Module):
    def __init__(self, dim, bias):
        super(FFTAttention, self).__init__()
        self.kernel_0_a = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_0_p = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_1_a = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_1_p = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_2_a = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_2_p = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_3_a = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.kernel_3_p = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4*dim, 4*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*dim//2, 4*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4*dim, 4*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*dim//2, 4*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.ca_conv1 = nn.Sequential(
            nn.Conv2d(4*dim, dim, 1),
        )
        self.ca_conv2 = nn.Sequential(
            nn.Conv2d(4*dim, dim, 1),
        )
    def forward(self, x,p_i):
        b,c,h,w = x.shape
        fft_x = torch.fft.fft2(x)
        p_i_0,p_i_1,p_i_2,p_i_3 = p_i.chunk(4, dim=1)
        p_i_0_real = self.kernel_0_a(p_i_0.real)
        p_i_0_imag = self.kernel_0_p(p_i_0.imag)
        p_i_0_real = p_i_0_real.view(-1,c*2,1,1)
        p_i_0_real1,p_i_0_real2=p_i_0_real.chunk(2, dim=1)
        x_0_real = fft_x.real*p_i_0_real1+p_i_0_real2
        p_i_0_imag = p_i_0_imag.view(-1,c*2,1,1)
        p_i_0_imag1,p_i_0_imag2=p_i_0_imag.chunk(2, dim=1)
        x_0_imag = fft_x.imag*p_i_0_imag1+p_i_0_imag2

        p_i_1_real = self.kernel_1_a(p_i_1.real)
        p_i_1_imag = self.kernel_1_p(p_i_1.imag)
        p_i_1_real = p_i_1_real.view(-1,c*2,1,1)
        p_i_1_real1,p_i_1_real2=p_i_1_real.chunk(2, dim=1)
        x_1_real = fft_x.real*p_i_1_real1+p_i_1_real2
        p_i_1_imag = p_i_1_imag.view(-1,c*2,1,1)
        p_i_1_imag1,p_i_1_imag2=p_i_1_imag.chunk(2, dim=1)
        x_1_imag = fft_x.imag*p_i_1_imag1+p_i_1_imag2

        p_i_2_real = self.kernel_2_a(p_i_2.real)
        p_i_2_imag = self.kernel_2_p(p_i_2.imag)
        p_i_2_real = p_i_2_real.view(-1,c*2,1,1)
        p_i_2_real1,p_i_2_real2=p_i_2_real.chunk(2, dim=1)
        x_2_real = fft_x.real*p_i_2_real1+p_i_2_real2
        p_i_2_imag = p_i_2_imag.view(-1,c*2,1,1)
        p_i_2_imag1,p_i_2_imag2=p_i_2_imag.chunk(2, dim=1)
        x_2_imag = fft_x.imag*p_i_2_imag1+p_i_2_imag2

        p_i_3_real = self.kernel_3_a(p_i_3.real)
        p_i_3_imag = self.kernel_3_p(p_i_3.imag)
        p_i_3_real = p_i_3_real.view(-1,c*2,1,1)
        p_i_3_real1,p_i_3_real2=p_i_3_real.chunk(2, dim=1)
        x_3_real = fft_x.real*p_i_3_real1+p_i_3_real2
        p_i_3_imag = p_i_3_imag.view(-1,c*2,1,1)
        p_i_3_imag1,p_i_3_imag2=p_i_3_imag.chunk(2, dim=1)
        x_3_imag = fft_x.imag*p_i_3_imag1+p_i_3_imag2

        x_real = torch.cat([x_0_real,x_1_real,x_2_real,x_3_real], dim=1)
        x_imag = torch.cat([x_0_imag,x_1_imag,x_2_imag,x_3_imag], dim=1)

        x_real = self.ca1(x_real) * x_real
        x_imag = self.ca2(x_imag) * x_imag

        x_real = self.ca_conv1(x_real)
        x_imag = self.ca_conv2(x_imag)
        x = x_real + 1j*x_imag
        ifft_x = torch.fft.ifft2(x)
        x = ifft_x.real
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.fft_ca = FFTAttention(dim, bias)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.msffn = MSFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        fft_p_i=y[1]
        p_i=y[2]
        p_i_img=y[3]
        x = x + self.fft_ca(self.norm3(x),fft_p_i)
        x = x + self.attn(self.norm1(x),p_i_img)
        x = x + self.msffn(self.norm4(x), p_i)
        x = x + self.ffn(self.norm2(x),p_i_img)

        return [x,fft_p_i,p_i,p_i_img]

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        #print("x.shape",x.shape)
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Priorformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(Priorformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down4_5 = Downsample(int(dim*2**1)) ## From Level 4 to Level 5
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, inp_img, p_i,p_i_img):
        
        p_i_0,p_i_1,p_i_2,p_i_3 = p_i.chunk(4, dim=1)
        fft_p_i_0 = torch.fft.fft2(p_i_0)
        fft_p_i_1 = torch.fft.fft2(p_i_1)
        fft_p_i_2 = torch.fft.fft2(p_i_2)
        fft_p_i_3 = torch.fft.fft2(p_i_3)
        fft_p_i = torch.cat([fft_p_i_0, fft_p_i_1, fft_p_i_2, fft_p_i_3], dim=1)
        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1,_,_,_ = self.encoder_level1([inp_enc_level1,fft_p_i,p_i,p_i_img])
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,_,_,_ = self.encoder_level2([inp_enc_level2,fft_p_i,p_i,p_i_img])
        
        inp_enc_level5 = self.down4_5(out_enc_level2)
        latent,_,_,_ = self.latent([inp_enc_level5,fft_p_i,p_i,p_i_img]) 
        
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2,_,_,_ = self.decoder_level2([inp_dec_level2,fft_p_i,p_i,p_i_img]) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,_,_,_ = self.decoder_level1([inp_dec_level1,fft_p_i,p_i,p_i_img])
        
        out_dec_level1,_,_,_ = self.refinement([out_dec_level1,fft_p_i,p_i,p_i_img])
        
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

class VQVAE_Prior(nn.Module):
    def __init__(self, patch_nums = (1, 4, 8, 16),vae_ckpt='your-path/VAR-main/vae_ch160v4096z32.pth'):
        super(VQVAE_Prior, self).__init__()
        self.vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        self.patch_nums = patch_nums
        # load checkpoints
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        #for para in self.vae.parameters():
            #para.requires_grad = False
        print(f'preparation finished.')
        #n_feats = 32
        n_feats_1 = 64

        VQE1_0=[nn.Conv2d(32, n_feats_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        VQE2_0=[
            common.ResBlock(
                common.default_conv, n_feats_1, kernel_size=3
            ) for _ in range(5)
        ]
        
        VQE3_0=[nn.Conv2d(n_feats_1, n_feats_1*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        
        self.ca0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats_1, n_feats_1//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1//2, n_feats_1, kernel_size=1),
            nn.Sigmoid()
        )

        self.sa0 = nn.Sequential(
            nn.Conv2d(n_feats_1, n_feats_1 // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.VQE4_0 = nn.Sequential(
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        VQE0_all = VQE1_0 + VQE2_0 + VQE3_0
        self.VQE0 = nn.Sequential(
            *VQE0_all
        )
        self.mlp_vq0 = nn.Sequential(
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True)
        )

        VQE1_1=[nn.Conv2d(32, n_feats_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        VQE2_1=[
            common.ResBlock(
                common.default_conv, n_feats_1, kernel_size=3
            ) for _ in range(5)
        ]

        VQE3_1=[nn.Conv2d(n_feats_1, n_feats_1*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        
        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats_1, n_feats_1//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1//2, n_feats_1, kernel_size=1),
            nn.Sigmoid()
        )

        self.sa1 = nn.Sequential(
            nn.Conv2d(n_feats_1, n_feats_1 // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.VQE4_1=nn.Sequential(
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        VQE1_all = VQE1_1 + VQE2_1 + VQE3_1
        self.VQE1 = nn.Sequential(
            *VQE1_all
        )
        self.mlp_vq1 = nn.Sequential(
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True)
        )

        VQE1_2=[nn.Conv2d(32, n_feats_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        VQE2_2=[
            common.ResBlock(
                common.default_conv, n_feats_1, kernel_size=3
            ) for _ in range(5)
        ]

        VQE3_2=[nn.Conv2d(n_feats_1, n_feats_1*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        
        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats_1, n_feats_1//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1//2, n_feats_1, kernel_size=1),
            nn.Sigmoid()
        )

        self.sa2 = nn.Sequential(
            nn.Conv2d(n_feats_1, n_feats_1 // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.VQE4_2 = nn.Sequential(
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        VQE2_all = VQE1_2 + VQE2_2 + VQE3_2
        self.VQE2 = nn.Sequential(
            *VQE2_all
        )
        self.mlp_vq2 = nn.Sequential(
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True)
        )

        VQE1_3=[nn.Conv2d(32, n_feats_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        VQE2_3=[
            common.ResBlock(
                common.default_conv, n_feats_1, kernel_size=3
            ) for _ in range(5)
        ]

        VQE3_3=[nn.Conv2d(n_feats_1, n_feats_1*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        
        self.ca3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats_1, n_feats_1//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1//2, n_feats_1, kernel_size=1),
            nn.Sigmoid()
        )

        self.sa3 = nn.Sequential(
            nn.Conv2d(n_feats_1, n_feats_1 // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.VQE4_3 = nn.Sequential(
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        VQE3_all = VQE1_3 + VQE2_3 + VQE3_3
        self.VQE3 = nn.Sequential(
            *VQE3_all
        )
        self.mlp_vq3 = nn.Sequential(
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True)
        )
        
        
        
        E1=[nn.Conv2d(32, n_feats_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats_1, kernel_size=3
            ) for _ in range(5)
        ]
        E3=[nn.Conv2d(n_feats_1, n_feats_1*2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats_1, n_feats_1//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1//2, n_feats_1, kernel_size=1),
            nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(n_feats_1, n_feats_1 // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats_1 // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.E4=nn.Sequential(
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats_1 * 2, n_feats_1 * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats_1 * 4, n_feats_1 * 4),
            nn.LeakyReLU(0.1, True)
        )
    def normalize_01_into_pm1(self, x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    
    def forward(self, gt_img_or):
        gt_img = gt_img_or.clone()
        
        gt_img_normal = self.normalize_01_into_pm1(gt_img)
        gt_img_tokens,encoder_fea = self.vae.img_to_idxBl(gt_img_normal, self.patch_nums)
        features_f_hat = []
        for index, img_token in enumerate(gt_img_tokens):
            gt_img_tokens_embedding = self.vae.quantize.embedding(img_token)
            gt_img_tokens_embedding_detach = gt_img_tokens_embedding.detach()
            gt_img_tokens_embedding_detach_clone = gt_img_tokens_embedding_detach.clone()
            
            features_f_hat.append(gt_img_tokens_embedding_detach_clone)
        
        
        b,hw,c = features_f_hat[0].shape
        h = w = int(math.sqrt(hw))
        features_f_hat_0 = rearrange(features_f_hat[0],'b (h w) c -> b c h w',h=h,w=w)
        
        b,hw,c = features_f_hat[1].shape
        h = w = int(math.sqrt(hw))
        features_f_hat_1 = rearrange(features_f_hat[1],'b (h w) c -> b c h w',h=h,w=w)
        b,hw,c = features_f_hat[2].shape
        h = w = int(math.sqrt(hw))
        features_f_hat_2 = rearrange(features_f_hat[2],'b (h w) c -> b c h w',h=h,w=w)
        b,hw,c = features_f_hat[3].shape
        h = w = int(math.sqrt(hw))
        features_f_hat_3 = rearrange(features_f_hat[3],'b (h w) c -> b c h w',h=h,w=w)

        fea_vq0 = self.VQE0(features_f_hat_0)
        fea_vq01,fea_vq02 = fea_vq0.chunk(2, dim=1)
        fea_vq01 = self.ca0(fea_vq01) * fea_vq01 
        b,c,h,w = fea_vq02.shape
        fea_vq02 = self.sa0(fea_vq02).view(b,h*w,1) * rearrange(fea_vq02, 'b c h w -> b (h w) c',h=h,w=w)
        fea_vq02 = rearrange(fea_vq02,'b (h w) c -> b c h w',h=h,w=w)
        fea_vq0 = self.VQE4_0(torch.cat([fea_vq01,fea_vq02], dim=1)).squeeze(-1).squeeze(-1)
        fea_vq0 = self.mlp_vq0(fea_vq0)

        fea_vq1 = self.VQE1(features_f_hat_1)
        fea_vq11,fea_vq12 = fea_vq1.chunk(2, dim=1)
        fea_vq11 = self.ca1(fea_vq11) * fea_vq11 
        b,c,h,w = fea_vq12.shape
        fea_vq12 = self.sa1(fea_vq12).view(b,h*w,1) * rearrange(fea_vq12, 'b c h w -> b (h w) c',h=h,w=w)
        fea_vq12 = rearrange(fea_vq12,'b (h w) c -> b c h w',h=h,w=w)
        fea_vq1 = self.VQE4_1(torch.cat([fea_vq11,fea_vq12], dim=1)).squeeze(-1).squeeze(-1)
        fea_vq1 = self.mlp_vq1(fea_vq1)

        fea_vq2 = self.VQE2(features_f_hat_2)
        fea_vq21,fea_vq22 = fea_vq2.chunk(2, dim=1)
        fea_vq21 = self.ca2(fea_vq21) * fea_vq21 
        b,c,h,w = fea_vq22.shape
        fea_vq22 = self.sa2(fea_vq22).view(b,h*w,1) * rearrange(fea_vq22, 'b c h w -> b (h w) c',h=h,w=w)
        fea_vq22 = rearrange(fea_vq22,'b (h w) c -> b c h w',h=h,w=w)
        fea_vq2 = self.VQE4_2(torch.cat([fea_vq21,fea_vq22], dim=1)).squeeze(-1).squeeze(-1)
        fea_vq2 = self.mlp_vq2(fea_vq2)

        fea_vq3 = self.VQE3(features_f_hat_3)
        fea_vq31,fea_vq32 = fea_vq3.chunk(2, dim=1)
        fea_vq31 = self.ca3(fea_vq31) * fea_vq31 
        b,c,h,w = fea_vq32.shape
        fea_vq32 = self.sa3(fea_vq32).view(b,h*w,1) * rearrange(fea_vq32, 'b c h w -> b (h w) c',h=h,w=w)
        fea_vq32 = rearrange(fea_vq32,'b (h w) c -> b c h w',h=h,w=w)
        fea_vq3 = self.VQE4_3(torch.cat([fea_vq31,fea_vq32], dim=1)).squeeze(-1).squeeze(-1)
        fea_vq3 = self.mlp_vq3(fea_vq3)

        features_p = [fea_vq0,fea_vq1,fea_vq2,fea_vq3]
        S1_prior_fea = torch.cat(features_p,dim=1)
        
        fea = self.E(encoder_fea)
        fea1,fea2 = fea.chunk(2, dim=1)
        fea1 = self.ca(fea1) * fea1 
        b,c,h,w = fea2.shape
        fea2 = self.sa(fea2).view(b,h*w,1) * rearrange(fea2, 'b c h w -> b (h w) c',h=h,w=w)
        fea2 = rearrange(fea2,'b (h w) c -> b c h w',h=h,w=w)
        fea = self.E4(torch.cat([fea1,fea2], dim=1)).squeeze(-1).squeeze(-1)
        fea = self.mlp(fea)
        return S1_prior_fea,fea
    
class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res
    
def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class ComplexMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ComplexMLP, self).__init__()
        self.resmlp = nn.Sequential(
            ComplexLinear(n_feats, n_feats),
            
        )
        self.relu =  nn.LeakyReLU(0.1, False)
    def complex_relu(self, input):
        return self.relu(input.real).type(torch.complex64)+1j*self.relu(input.imag).type(torch.complex64)
    def forward(self, x):
        res=self.resmlp(x)
        res = self.complex_relu(res)
        return res

class denoise1(nn.Module):
    def __init__(self,n_feats = 64,timesteps=5):
        super(denoise1, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=1*n_feats
        
        self.relu =  nn.LeakyReLU(0.1, False)
        resmlp = [
            ComplexLinear(n_featsx4*2+1, n_featsx4),
            
        ]
        
        self.resmlp=nn.Sequential(*resmlp)
        self.resmlp1=ComplexMLP(n_featsx4)
        
        
    def complex_relu(self, input):
        return self.relu(input.real).type(torch.complex64)+1j*self.relu(input.imag).type(torch.complex64)
    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        fea = self.resmlp(c)
        
        fea = self.complex_relu(fea)
        
        fea = self.resmlp1(fea)
        
        return fea 

class denoise(nn.Module):
    def __init__(self,n_feats = 64,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=1*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(1):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        
        fea = self.resmlp(c)

        return fea 
    


@ARCH_REGISTRY.register()
class MSRestoreXS2(nn.Module):
    def __init__(self,     
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        linear_start= 0.1,
        linear_end= 0.99, 
        timesteps = 4,
        patch_nums = (16, 32, 64, 128),
        vae_ckpt='your-path/VAR-main/vae_ch160v4096z32.pth',
        ):
        super(MSRestoreXS2, self).__init__()

        # Generator
        self.G = Priorformer(        
        inp_channels=inp_channels, 
        out_channels=out_channels, 
        dim = dim,
        num_blocks = num_blocks, 
        num_refinement_blocks = num_refinement_blocks,
        heads = heads,
        ffn_expansion_factor = ffn_expansion_factor,
        bias = bias,
        LayerNorm_type = LayerNorm_type   ## Other option 'BiasFree'
        )
        
        self.condition = VQVAE_Prior(patch_nums=patch_nums,vae_ckpt=vae_ckpt)
        
        self.denoise= denoise(n_feats=1024, timesteps=timesteps)
        self.diffusion = DDPM2(denoise=self.denoise, condition=self.condition ,n_feats=256,linear_start= linear_start,linear_end= linear_end, timesteps = timesteps)
        
        self.condition_img = VQVAE_Prior(patch_nums=patch_nums,vae_ckpt=vae_ckpt)
        self.denoise_img = denoise(n_feats=256, timesteps=timesteps)
        self.diffusion_img = DDPM(denoise=self.denoise_img, condition=self.condition_img ,n_feats=64,linear_start= linear_start,linear_end= linear_end, timesteps = timesteps)
        
        self.padding_width = nn.ConstantPad2d((0,1),0)
        self.padding_height = nn.ConstantPad2d((0,0,1,0),0)
    def forward(self, img, S1_prior_fea=None, S1_prior_fea_img=None):
        if self.training:
            S2_prior_fea, _ = self.diffusion(img,S1_prior_fea)
            S2_prior_fea_img, _ = self.diffusion_img(img,S1_prior_fea_img)
            
            sr = self.G(img, S2_prior_fea,S2_prior_fea_img)
            return sr, S2_prior_fea,S2_prior_fea_img
        else:
            
            padding_num_h = 0
            padding_num_w = 0
            n,c,h,w = img.shape
            while (w/8)%2 != 0:
                img = self.padding_width(img)
                padding_num_w = padding_num_w + 1
                w = w + 1
            while (h/8)%2 != 0:
                img = self.padding_height(img)
                padding_num_h = padding_num_h + 1
                h = h + 1
            
            S2_prior_fea=self.diffusion(img)
            S2_prior_fea_img=self.diffusion_img(img)
            sr = self.G(img, S2_prior_fea,S2_prior_fea_img)
            if padding_num_w!=0:
                sr = sr[:,:,:,:-padding_num_w]
                
            if padding_num_h!=0:
                sr = sr[:,:,padding_num_h:,:]
            
            return sr
