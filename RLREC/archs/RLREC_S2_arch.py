from ldm.ddpm import WLRRDM
import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
import torchvision.transforms.functional as TF


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


class FeedForward_cross(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_cross, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_in2 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features*2, dim, kernel_size=1, bias=bias)

        self.kernel1 = nn.Sequential(nn.Linear(256, dim*2, bias=False),)
    def forward(self, x,lq_d_gt_c,lq_c_gt_d):
        b,c,h,w = x.shape
        lq_d_gt_c=self.kernel1(lq_d_gt_c).view(-1,c*2,1,1)
        lq_d_gt_c1,lq_d_gt_c2=lq_d_gt_c.chunk(2, dim=1)
        x1 = x*lq_d_gt_c1+lq_d_gt_c2

        lq_c_gt_d=self.kernel1(lq_c_gt_d).view(-1,c*2,1,1)
        lq_c_gt_d1,lq_c_gt_d2=lq_c_gt_d.chunk(2, dim=1)
        x2 = x*lq_c_gt_d1+lq_c_gt_d2

        x1 = self.project_in1(x1)
        x11, x12 = self.dwconv1(x1).chunk(2, dim=1)
        x2 = self.project_in2(x2)
        x21, x22 = self.dwconv2(x2).chunk(2, dim=1)
        x1 = F.gelu(x11) * x22
        x2 = F.gelu(x21) * x12
        x = self.project_out(torch.cat([x1,x2],dim=1))
        return x

class Attention_cross(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_cross, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads//2, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads//2, 1, 1))
        self.kernel1 = nn.Sequential(nn.Linear(256, dim//num_heads*6, bias=False),)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x,lq_d_gt_c,lq_c_gt_d):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #print("q.shape", q.shape)
        q1, q2 = q.chunk(2, dim=1)
        k1, k2 = k.chunk(2, dim=1)
        v1, v2 = v.chunk(2, dim=1)
        
        lq_d_gt_c = self.kernel1(lq_d_gt_c).view(-1,1,c//self.num_heads*6,1)
        lq_d_gt_c11,lq_d_gt_c12,lq_d_gt_c13,lq_d_gt_c21,lq_d_gt_c22,lq_d_gt_c23 = lq_d_gt_c.chunk(6, dim=2)
        q1 = q1*lq_d_gt_c11+lq_d_gt_c21
        k1 = k1*lq_d_gt_c12+lq_d_gt_c22
        v1 = v1*lq_d_gt_c13+lq_d_gt_c23
        
        lq_c_gt_d = self.kernel1(lq_c_gt_d).view(-1,1,c//self.num_heads*6,1)
        lq_c_gt_d11,lq_c_gt_d12,lq_c_gt_d13,lq_c_gt_d21,lq_c_gt_d22,lq_c_gt_d23 = lq_c_gt_d.chunk(6, dim=2)
        q2 = q2*lq_c_gt_d11+lq_c_gt_d21
        k2 = k2*lq_c_gt_d12+lq_c_gt_d22
        v2 = v2*lq_c_gt_d13+lq_c_gt_d23

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature1
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v2)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature2
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v1)

        out = torch.cat([out1,out2],dim=1)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        
        return out
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_cross = Attention_cross(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_cross = FeedForward_cross(dim, ffn_expansion_factor, bias)
        
        
    def forward(self, y):
        x = y[0]
        lq_d_gt_c=y[1]
        lq_c_gt_d=y[2]
        x = x + self.attn_cross(self.norm1(x),lq_d_gt_c,lq_c_gt_d)
        x = x + self.ffn_cross(self.norm2(x),lq_d_gt_c,lq_c_gt_d)
        return [x,lq_d_gt_c,lq_c_gt_d]
    
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
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class RLRformer(nn.Module):
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

        super(RLRformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) 
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) 
        
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        

    def forward(self, inp_img, lq_d_gt_c,lq_c_gt_d):    
        inp_enc_level1 = self.patch_embed(inp_img)
        
        out_enc_level1,_,_ = self.encoder_level1([inp_enc_level1,lq_d_gt_c,lq_c_gt_d])
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,_,_ = self.encoder_level2([inp_enc_level2,lq_d_gt_c,lq_c_gt_d])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        
        latent,_,_ = self.latent([inp_enc_level3,lq_d_gt_c,lq_c_gt_d])
        
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2,_,_ = self.decoder_level2([inp_dec_level2,lq_d_gt_c,lq_c_gt_d]) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,_,_ = self.decoder_level1([inp_dec_level1,lq_d_gt_c,lq_c_gt_d])
        
        out_dec_level1,_,_ = self.refinement([out_dec_level1,lq_d_gt_c,lq_c_gt_d])
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1

class dino_A(nn.Module):
    def __init__(self, n_feats = 64, DINO_REPO_DIR=None):
        super(dino_A, self).__init__()
        
        self.dino_mlp_2 = nn.Sequential(
            nn.Linear(32, 24),
            nn.LeakyReLU(0.1, True),
            nn.Linear(24, 12),
            nn.LeakyReLU(0.1, True),
            nn.Linear(12, 1),
            nn.LeakyReLU(0.1, True),
        )

        self.dino_mlp = nn.Sequential(
            nn.Linear(n_feats * 12 * 6, n_feats * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 8, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            
        )
        
        self.dinov3_vitb16 = torch.hub.load(DINO_REPO_DIR, 'dinov3_vitb16', source='local', weights='dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')

        
    # image resize transform to dimensions divisible by patch size
    def resize_transform(self,img,IMAGENET_MEAN=(0.485, 0.456, 0.406),IMAGENET_STD=(0.229, 0.224, 0.225)):
        image_resized = TF.resize(img, (512, 512))
        image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image_resized_norm
    
    def forward(self, img):
        b,c,h,w = img.shape
        x = self.resize_transform(img)
        x_feats = self.dinov3_vitb16.get_intermediate_layers(x, n=range(12), reshape=True, norm=True)
        x_feats_last = torch.cat([x_feats[0],x_feats[1],x_feats[2],x_feats[3],x_feats[4],x_feats[5]],dim=1)
        x_feats_last = self.dino_mlp_2(x_feats_last.detach())
        x_feats_last = self.dino_mlp_2(x_feats_last.squeeze(-1))
        output = self.dino_mlp(x_feats_last.squeeze(-1))
        return output


class dino_S(nn.Module):
    def __init__(self, n_feats = 64, DINO_REPO_DIR=None):
        super(dino_S, self).__init__()
        
        self.dino_mlp_2 = nn.Sequential(
            nn.Linear(32, 24),
            nn.LeakyReLU(0.1, True),
            nn.Linear(24, 12),
            nn.LeakyReLU(0.1, True),
            nn.Linear(12, 1),
            nn.LeakyReLU(0.1, True),
        )

        self.dino_mlp = nn.Sequential(
            nn.Linear(n_feats * 12 * 6, n_feats * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 8, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            
        )
        
        self.dinov3_vits16 = torch.hub.load(DINO_REPO_DIR, 'dinov3_vitb16', source='local', weights='dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
        
    
    def resize_transform(self,img,IMAGENET_MEAN=(0.485, 0.456, 0.406),IMAGENET_STD=(0.229, 0.224, 0.225)):
        image_resized = TF.resize(img, (512, 512))
        image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image_resized_norm
    
    def forward(self, img):
        b,c,h,w = img.shape
        x = self.resize_transform(img)
        x_feats = self.dinov3_vits16.get_intermediate_layers(x, n=range(12), reshape=True, norm=True)
        x_feats_last = torch.cat([x_feats[6],x_feats[7],x_feats[8],x_feats[9],x_feats[10],x_feats[11]],dim=1)
        x_feats_last = self.dino_mlp_2(x_feats_last.detach())
        x_feats_last = self.dino_mlp_2(x_feats_last.squeeze(-1))
        output = self.dino_mlp(x_feats_last.squeeze(-1))
        return output


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
    
    
class denoise(nn.Module):
    def __init__(self,n_feats = 64,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
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
class RLRECS2(nn.Module):
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
        DINO_REPO_DIR=None,
        linear_start= 0.1,
        linear_end= 0.99, 
        timesteps = 4 ):
        super(RLRECS2, self).__init__()

        # Generator
        self.G = RLRformer(        
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
        
        self.condition_wA = dino_A(n_feats=64,DINO_REPO_DIR=DINO_REPO_DIR)
        self.condition_wS = dino_S(n_feats=64,DINO_REPO_DIR=DINO_REPO_DIR)
        self.denoise= denoise(n_feats=64*2, timesteps=timesteps)
        self.diffusion = WLRRDM(denoise=self.denoise, condition=self.condition_wA, condition2=self.condition_wS, n_feats=64,linear_start=linear_start,linear_end=linear_end, timesteps=timesteps)
        self.padding_width = nn.ConstantPad2d((0,1),0)
        self.padding_height = nn.ConstantPad2d((0,0,1,0),0)
        
    def forward(self, img, S1_prior_fea1=None, S1_prior_fea2=None):
        if self.training:
            S2_prior_fea = self.diffusion(img,S1_prior_fea1, S1_prior_fea2)
            S2_prior_fea_wA,S2_prior_fea_wS = S2_prior_fea.chunk(2, dim=1)
            
            sr = self.G(img, S2_prior_fea_wA, S2_prior_fea_wS)
            return sr, S2_prior_fea_wA, S2_prior_fea_wS
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
            
            S2_prior_fea = self.diffusion(img)
            S2_prior_fea_wA,S2_prior_fea_wS = S2_prior_fea.chunk(2, dim=1)
            
            sr = self.G(img, S2_prior_fea_wA, S2_prior_fea_wS)
            if padding_num_w!=0:
                sr = sr[:,:,:,:-padding_num_w]
                
            if padding_num_h!=0:
                sr = sr[:,:,padding_num_h:,:]
                
            return sr
