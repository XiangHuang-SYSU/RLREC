import torch
from torch import nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
import torchvision.transforms.functional as TF

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
        
        loss_abs = 0
        #for i in range(len(S1_fea)):
        loss_abs += nn.L1Loss()(S2_fea, S1_fea.detach())
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
        
        loss_abs = 0
        #for i in range(len(S1_fea)):
        loss_abs += nn.L1Loss()(S2_fea, S1_fea.detach())
        return self.loss_weight * loss_abs


@LOSS_REGISTRY.register()
class DINOLoss(nn.Module):
    def __init__(self, loss_weight1=1.0, loss_weight2=1.0, DINO_REPO_DIR=None):
        super(DINOLoss, self).__init__()
        self.loss_weight1 = loss_weight1
        self.loss_weight2 = loss_weight2
        
        self.dinov3_vits16 = torch.hub.load(DINO_REPO_DIR, 'dinov3_vitb16', source='local', weights='dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    
    # image resize transform to dimensions divisible by patch size
    def resize_transform(self,img,IMAGENET_MEAN=(0.485, 0.456, 0.406),IMAGENET_STD=(0.229, 0.224, 0.225)):
        image_resized = TF.resize(img, (512, 512))
        image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image_resized_norm
    
    def forward(self, generated, real_image):
        x_generated = self.resize_transform(generated)
        gt_real_image = self.resize_transform(real_image)
        x_feats = self.dinov3_vits16.get_intermediate_layers(x_generated, n=range(12), reshape=True, norm=True)
        
        x_feats_fea_Appearance = torch.cat([x_feats[0],x_feats[1],x_feats[2],x_feats[3],x_feats[4],x_feats[5]],dim=1)
        x_feats_fea_Structure = torch.cat([x_feats[6],x_feats[7],x_feats[8],x_feats[9],x_feats[10],x_feats[11]],dim=1)

        gt_feats = self.dinov3_vits16.get_intermediate_layers(gt_real_image, n=range(12), reshape=True, norm=True)
        gt_feats_fea_Appearance = torch.cat([gt_feats[0],gt_feats[1],gt_feats[2],gt_feats[3],gt_feats[4],gt_feats[5]],dim=1)
        gt_feats_fea_Structure = torch.cat([gt_feats[6],gt_feats[7],gt_feats[8],gt_feats[9],gt_feats[10],gt_feats[11]],dim=1)

        loss_abs_Appearance = nn.L1Loss()(x_feats_fea_Appearance, gt_feats_fea_Appearance)
        loss_abs_Structure = nn.L1Loss()(x_feats_fea_Structure, gt_feats_fea_Structure)
        return self.loss_weight1 * loss_abs_Appearance, self.loss_weight2 * loss_abs_Structure
        




