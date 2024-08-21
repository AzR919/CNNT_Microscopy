"""
Losses used for CNNT training
"""

import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics_ssim3d import StructuralSimilarityIndexMeasure3D
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanAbsoluteError

import torch.nn as nn
import torch.nn.functional as F

import pytorch_ssim

# -------------------------------------------------------------------------------------------------

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class Weighted_Sobel_Complex_Loss:
    """
    Weighted loss for complex with Sobel operator
    """
    def __init__(self, device='cpu'):
        self.sobel = Sobel()
        self.sobel.to(device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape

        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.reshape(outputs_im, (B*T, 1, H, W))
        targets_im = torch.reshape(targets_im, (B*T, 1, H, W))
        diff_sobel_square = torch.square(self.sobel(outputs_im)-self.sobel(targets_im))

        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            v_sobel = torch.sum(weights_used[:, None, None, None] * diff_sobel_square) / torch.sum(weights_used)
        else:
            v_sobel = torch.sum(diff_sobel_square)

        if(torch.any(torch.isnan(v_sobel))):
            v_sobel = 0.0

        return torch.sqrt(v_sobel) / (B*T) / (H*W/4096)
        #return torch.sqrt(v_sobel) / (B*T) / (H*W)
        #return torch.sqrt( v_sobel / torch.numel(outputs) )

# -------------------------------------------------------------------------------------------------

class Weighted_SSIM_Complex_Loss_Prev:
    """
    Weighted loss for complex with SSIM
    """
    def __init__(self, reduction='mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.ssim_loss = pytorch_ssim.SSIM(size_average=False, window_size=window_size, device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.reshape(outputs_im, (B*T, 1, H, W))
        targets_im = torch.reshape(targets_im, (B*T, 1, H, W))
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = 1.0

        return (1.0-v_ssim)

class Weighted_SSIM_Complex_Loss:
    """
    Weighted loss for complex with SSIM
    """
    def __init__(self, reduction='elementwise_mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=window_size, reduction=self.reduction).to(device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.reshape(outputs_im, (B*T, 1, H, W))
        targets_im = torch.reshape(targets_im, (B*T, 1, H, W))
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = 1.0

        return (1.0-v_ssim)

class Weighted_MSSSIM_Complex_Loss:
    """
    Weighted loss for complex with SSIM
    """
    def __init__(self, reduction='elementwise_mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.device=device 
        self.ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=window_size, reduction=self.reduction).to(device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.reshape(outputs_im, (B*T, 1, H, W))
        targets_im = torch.reshape(targets_im, (B*T, 1, H, W))
        
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = torch.tensor(1.0).to(self.device)

        return (1.0-v_ssim)
    
# -------------------------------------------------------------------------------------------------

class Weighted_SSIM3D_Complex_Loss:
    """
    Weighted loss for complex with SSIM3D
    """
    def __init__(self, reduction='elementwise_mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.ssim_loss = StructuralSimilarityIndexMeasure3D(data_range=1.0, kernel_size=window_size, reduction=self.reduction).to(device) #pytorch_ssim.SSIM3D(size_average=False, window_size=window_size, device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.permute(outputs_im, (0, 2, 1, 3, 4))
        targets_im = torch.permute(targets_im, (0, 2, 1, 3, 4))
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = 1.0

        return (1.0-v_ssim)

class Weighted_SSIM3D_Complex_Loss_Prev:
    """
    Weighted loss for complex with SSIM3D
    """
    def __init__(self, reduction='mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.ssim_loss = pytorch_ssim.SSIM3D(size_average=False, window_size=window_size, device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.permute(outputs_im, (0, 2, 1, 3, 4))
        targets_im = torch.permute(targets_im, (0, 2, 1, 3, 4))
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = 1.0

        return (1.0-v_ssim)

# -------------------------------------------------------------------------------------------------

class Weighted_L1_Complex_Loss_Prev:
    """
    Weighted L1 loss for complex
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape

        if(C==2):
            diff_L1 = torch.abs(outputs[:,:,0,:,:]-targets[:,:,0,:,:]) + torch.abs(outputs[:,:,1,:,:]-targets[:,:,1,:,:])
        else:
            diff_L1 = torch.abs(outputs-targets)

        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                v_l1 = torch.sum(weights.reshape(B,T,1,1,1) * diff_L1.reshape(B,T,1,H,W)) / torch.sum(weights)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                v_l1 = torch.sum(weights.reshape(B,1,1,1,1) * diff_L1.reshape(B,T,1,H,W)) / torch.sum(weights)
        else:
            v_l1 = torch.sum(diff_L1.reshape(B,T,1,H,W))

        return v_l1 / (B*T) / (H*W/4096)

class Weighted_L1_Complex_Loss:
    """
    Weighted L1 loss for complex
    """
    def __init__(self, reduction='mean', device='cpu'):
        self.reduction=reduction
        self.l1_loss = nn.L1Loss(reduction=self.reduction).to(device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape

        if(C==2):
            l1 = self.l1_loss(outputs[:,:,0,:,:], targets[:,:,0,:,:]) + self.l1_loss(outputs[:,:,1,:,:], targets[:,:,1,:,:])
        else:
            l1 = self.l1_loss(outputs, targets) 
        return l1
    
# -------------------------------------------------------------------------------------------------

class Weighted_MSE_Complex_Loss_Prev:
    """
    Weighted MSE loss for complex
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            diff_mag_square = torch.square(outputs[:,:,0,:,:]-targets[:,:,0,:,:]) + torch.square(outputs[:,:,1,:,:]-targets[:,:,1,:,:])
        else:
            diff_mag_square = torch.square(outputs-targets)

        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                v_l2 = torch.sum(weights.reshape(B,T,1,1,1) * diff_mag_square.reshape(B,T,1,H,W)) / torch.sum(weights)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                v_l2 = torch.sum(weights.reshape(B,1,1,1,1) * diff_mag_square.reshape(B,T,1,H,W)) / torch.sum(weights)
        else:
            v_l2 = torch.sum(diff_mag_square.reshape(B,T,1,H,W))

        return torch.sqrt(v_l2) / (B*T) / (H*W/4096)

class Weighted_MSE_Complex_Loss:
    """
    Weighted MSE loss for complex
    """
    def __init__(self, reduction='mean', device='cpu'):
        self.reduction=reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction).to(device)

    def __call__(self, outputs, targets):

        B, T, C, H, W = targets.shape
        if(C==2):
            mse = self.mse_loss(outputs[:,:,0,:,:], targets[:,:,0,:,:]) + self.mse_loss(outputs[:,:,1,:,:], targets[:,:,1,:,:])
        else:
            mse = self.mse_loss(outputs, targets)

        return mse
    
class MSE_Complex_Loss_Prev:
    """
    MSE loss for complex
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets):

        B, T, C, H, W = targets.shape
        if(C==2):
            diff_mag_square = torch.square(outputs[:,:,0,:,:]-targets[:,:,0,:,:]) + torch.square(outputs[:,:,1,:,:]-targets[:,:,1,:,:])
        else:
            diff_mag_square = torch.square(outputs-targets)

        v_l2 = torch.sum(diff_mag_square)

        return torch.sqrt(v_l2) / (B*T) / (H*W/4096)

class MSE_Complex_Loss:
    """
    MSE loss for complex
    """
    def __init__(self, reduction='mean', device='cpu'):
        self.reduction=reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction).to(device)

    def __call__(self, outputs, targets):

        B, T, C, H, W = targets.shape
        if(C==2):
            mse = self.mse_loss(outputs[:,:,0,:,:], targets[:,:,0,:,:]) + self.mse_loss(outputs[:,:,1,:,:], targets[:,:,1,:,:])
        else:
            mse = self.mse_loss(outputs, targets)

        return mse

# -------------------------------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets, weights=None):
        diff = outputs - targets
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()
    
# ------------------------------------------------------------------------------------------------
    
class PSNR_Prev:
    """
    PSNR for metric comparison
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets):

        return -4.342944819 * torch.log(torch.mean(torch.square(targets - outputs)))
    
class PSNR:
    """
    PSNR for metric comparison
    """
    def __init__(self, reduction='elementwise_mean', device='cpu'):
        self.reduction=reduction
        self.psnr_func = PeakSignalNoiseRatio(data_range=1.0, reduction=self.reduction).to(device)

    def __call__(self, outputs, targets):

        return self.psnr_func(outputs, targets)

# -------------------------------------------------------------------------------------------------
class BrightnessContrastLoss(nn.Module):
    def __init__(self):
        super(BrightnessContrastLoss, self).__init__()

    def forward(self, outputs, targets, weights=None):
        # Mean brightness
        outputs = outputs.permute(0,2,1,3,4)
        targets = targets.permute(0,2,1,3,4)
        brightness_output = outputs.mean([2, 3, 4])
        brightness_target = targets.mean([2, 3, 4])
        
        # Standard deviation for contrast
        contrast_output = outputs.std([2, 3, 4])
        contrast_target = targets.std([2, 3, 4])
        
        # Brightness loss
        brightness_loss = F.mse_loss(brightness_output, brightness_target)
        
        # Contrast loss
        contrast_loss = F.mse_loss(contrast_output, contrast_target)
        
        # Combine the losses
        total_loss = brightness_loss + contrast_loss
        return total_loss
    
class Image_Enhancement_Combined_Loss:
    """Combined loss for image enhancement
    """
    def __init__(self):
        self.losses = []

    def add_loss(self, a_loss, w=1.0):
        self.losses.append((a_loss, w))

    def __call__(self, outputs, targets, weights=None): # adding epoch here to keep code consistent

        assert len(self.losses) > 0

        combined_loss = self.losses[0][1] * self.losses[0][0](outputs=outputs, targets=targets, weights=weights)

        for k in range(1,len(self.losses)):
            combined_loss += self.losses[k][1] * self.losses[k][0](outputs=outputs, targets=targets, weights=weights)

        return combined_loss

    def __str__(self):
        content = f"Image_Enhancement_Combined_Loss, {len(self.losses)} losses\n"
        for l in self.losses:
            content += f"loss - {type(l[0])}, weights {l[1]}\n"

        return content

# -------------------------------------------------------------------------------------------------
