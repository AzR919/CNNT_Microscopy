"""
Convolutional Neural Net Transformer models

"""

import os
import sys
import json
import logging
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

from model import *
#from model_ms_cnnt import *
#from models.models_msunet3d import MSUNet3D
from enhancement_loss import *

# -------------------------------------------------------------------------------------------------

class CNNT_base_model_runtime(nn.Module):
    """CNNT base model for image enhancement
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.height = config.height[0]
        self.width = config.width[0]

    def set_up_scheduling(self, train_total_len):
        if self.config.optim is not None:

            if (self.config.optim  == "adamw"):
                if self.config.no_w_decay:
                    self.optim = self.configure_optimizers(self.config)
                else:
                    self.optim = optim.AdamW(self.parameters(), lr=self.config.global_lr, betas=(self.config.beta1, self.config.beta2), eps=1e-08,
                                            weight_decay=self.config.weight_decay, amsgrad=False)

            if (self.config.optim  == "sgd"):
                self.optim = optim.SGD(self.parameters(), lr=self.config.global_lr, momentum=0.9, weight_decay=self.config.weight_decay,
                                    nesterov=False)

            if (self.config.optim  == "nadam"):
                self.optim = optim.NAdam(self.parameters(), lr=self.config.global_lr, betas=(self.config.beta1, self.config.beta2), eps=1e-08,
                                        weight_decay=self.config.weight_decay, momentum_decay=0.004)

            # set up the scheduler
            self.scheduler = None

            if (self.config.scheduler == "ReduceLROnPlateau"):
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',
                                                                    patience=5,
                                                                    min_lr=[2e-6, self.config.global_lr] if self.config.no_w_decay else 2e-6,
                                                                    cooldown=2,
                                                                    factor=0.8,
                                                                    verbose=True)
                self.scheduler_on_batch = False

            if (self.config.scheduler == "StepLR"):
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 5, gamma=0.8, last_epoch=-1, verbose=True)
                self.scheduler_on_batch = False

            if (self.config.scheduler == "OneCycleLR"):
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.config.global_lr,
                                                                total_steps=train_total_len,
                                                                pct_start=0.3,
                                                                anneal_strategy='cos', cycle_momentum=True,
                                                                base_momentum=0.85, max_momentum=0.95,
                                                                div_factor=25,
                                                                final_div_factor=10000,
                                                                three_phase=False,
                                                                last_epoch=-1)

                self.scheduler_on_batch = True
        else:
            # Used for testing/plotting
            pass

    def configure_optimizers(self, config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Conv3d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.global_lr, betas=(self.config.beta1, self.config.beta2), eps=1e-08, amsgrad=False)

        return optimizer

    def set_up_loss(self, loss, loss_weights, device='cpu'):

        loss_f = Image_Enhancement_Combined_Loss()
        for ind, l in enumerate(loss):
            if(l == "mse"):
                loss_f.add_loss(Weighted_MSE_Complex_Loss(reduction='mean', device=device), w=loss_weights[ind])
            
            elif(l == 'charbonnier'):
                loss_f.add_loss(CharbonnierLoss().to(device), w=loss_weights[ind])
                                
            elif(l == "l1"):
                loss_f.add_loss(Weighted_L1_Complex_Loss(reduction='mean', device=device), w=loss_weights[ind])

            elif(l == "ssim"): #elementwise_mean
                loss_f.add_loss(Weighted_SSIM_Complex_Loss(reduction='elementwise_mean', window_size=7, device=device), w=loss_weights[ind])
            
            elif(l == "msssim"):
                loss_f.add_loss(Weighted_MSSSIM_Complex_Loss(reduction='elementwise_mean', window_size=7, device=device), w=loss_weights[ind])

            elif(l == "ssim3D"):
                loss_f.add_loss(Weighted_SSIM3D_Complex_Loss_Prev(reduction='mean', window_size=7, device=device), w=loss_weights[ind])

            elif(l == "sobel"):
                loss_f.add_loss(Weighted_Sobel_Complex_Loss(device=device), w=loss_weights[ind])
            
            elif(l == "bcl"):
                loss_f.add_loss(BrightnessContrastLoss().to(device), w=loss_weights[ind])

            else:
                raise f"loss type not supported:{l}"

        return loss_f

    def save(self, epoch):
        """
        Save the current model weights on the given epoch
        """
        save_file_name = f"{self.config.date}_epoch-{epoch}.pth"
        save_file_path = os.path.join(self.config.check_path, save_file_name)
        torch.save(self.state_dict(), save_file_path)

    def load(self, load_path, device=None):
        """
        Load a checkpoint is the load path is given in config
        """
        logging.info(f"Loading model from {load_path}")
        if(device is None):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(load_path, map_location=device), strict=False)

# -------------------------------------------------------------------------------------------------

def generate_model_file_name(config):
    """
    Use the config info to create meaningful model name

    Returns:
        str: model file string
    """

    model_str = config.run_name
    model_file_name = f"{model_str}__{config.date}_epoch-{config.num_epochs}"

    return model_file_name

def save_cnnt_model(model, config, last="", height=64, width=64):

    model.eval()
    C = 1

    model_input = torch.randn(1, config.time, C, height, width, requires_grad=True)
    model_input = model_input.to('cpu')
    model.to('cpu')

    model_str = generate_model_file_name(config)
    model_file_name = os.path.join(config.model_path, model_str)
    model_file_name += last

    logging.info(f"Saving model weights at: {model_file_name}.pt")
    torch.save(model.state_dict(), f"{model_file_name}.pt")
    logging.info(f"Model weights saved")

    if not config.train_only:
        if(config.norm_mode=="instance" or config.norm_mode=="mixed"):
            logging.info(f"Exporting to onnx does not work for norm_mode : {config.norm_mode}")
        else:
            logging.info(f"Saving onnx model at: {model_file_name}.onnx")

            if(config.norm_mode=="layer"):
                dynamic_axes_value = {0 : 'batch_size', 1 : 'time'}
            else:
                dynamic_axes_value = {0 : 'batch_size', 1 : 'time', 3 : 'H', 4 : 'W'}

            logging.info(f"dynamic_axes_value is: {dynamic_axes_value}")

            # Export the model
            torch.onnx.export(model,                   # model being run
                            model_input,               # model input (or a tuple for multiple inputs)
                            f"{model_file_name}.onnx", # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=11,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['image'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'image' :    # variable length axes
                                            dynamic_axes_value,
                                            'output' :
                                            dynamic_axes_value}
                            )

            logging.info(f"Onnx model saved")

        logging.info(f"Saving torchscript model at: {model_file_name}.pts")
        model_scripted = torch.jit.trace(model, (torch.rand(1, config.time, C, height, width)), strict=False)
        model_scripted.save(f"{model_file_name}.pts")
        logging.info(f"Torchscript model saved")

    with open(f"{model_file_name}.json", "w") as file:
        json.dump(dict(config), file)

    logging.info(f"Saved config as {model_file_name}.json")

    return model_file_name

# -------------------------------------------------------------------------------------------------

class CNNT_denoising_runtime(CNNT_base_model_runtime):
    """
    The full CNN_Transformer model
    """

    def __init__(self, config, train_total_len):
        super().__init__(config=config)

        K = config.kernel_size
        S = config.stride
        P = config.padding
        D = config.dropout_p
        with_mixer = (config.with_mixer>0)

        C_in = 1
        C_out = 1

        self.cnnt = CNNTUnet(blocks=config.blocks,
                         blocks_per_set=config.blocks_per_set,
                         H=self.height, W=self.width,
                         C_in=C_in,
                         T=config.time,
                         C_out=C_out,
                         n_head=config.n_head,
                         norm_mode=config.norm_mode,
                         kernel_size=(K,K), stride=(S,S), padding=(P,P),
                         dropout_p=D, with_mixer=with_mixer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # setup loss function and optimizer
        self.loss_f = self.set_up_loss(config.loss_DN, config.loss_DN_weights, device=device, ct_weight_loss=config.ct_weight_loss)

        self.set_up_scheduling(train_total_len=train_total_len)

        # if a load checkpoint is given, load it
        if config.load_path != None:
            self.load(config.load_path)


    def forward(self, x):
        # Pass the input to CNNT and work with the output

        noise = self.cnnt(x)
        output = x - noise

        return output

    def compute_loss(self, output, targets, weights, inputs=None):
        # compute loss
        loss = self.loss_f(output, targets, weights, inputs)
        return loss

# -------------------------------------------------------------------------------------------------

class Batch_perm(torch.nn.Module):
    def forward(self, x):
        return torch.permute(x, (0, 2, 1, 3, 4))

class CNNT_enhanced_denoising_runtime(CNNT_base_model_runtime):
    """
    The full CNN_Transformer model
    """

    def __init__(self, config, train_total_len):
        super().__init__(config=config)

        K = config.kernel_size
        S = config.stride
        P = config.padding
        D = config.dropout_p
        with_mixer = (config.with_mixer>0)

        self.no_residual = config.no_residual

        C_in = 1
        C_out = 1

        self.pre_cnnt = nn.Sequential(
            Conv2DExtOrg(C_in, 16, kernel_size=K, stride=S, padding=P, bias=True),
            Conv2DExtOrg(16, 32, kernel_size=K, stride=S, padding=P, bias=True)
        )
        #self.cnnt = MSUNet3D(in_channels=32, out_channels=32)
        self.cnnt = CNNTUnet(blocks=config.blocks,
                         blocks_per_set=config.blocks_per_set,
                         H=self.height, W=self.width,
                         C_in=32,
                         T=config.time,
                         C_out=32,
                         n_head=config.n_head,
                         norm_mode=config.norm_mode,
                         kernel_size=(K,K), stride=(S,S), padding=(P,P),
                         dropout_p=D, with_mixer=with_mixer)
        

        self.pos_cnnt = nn.Sequential(
            Conv2DExtOrg(32,    16, kernel_size=K, stride=S, padding=P, bias=True),
            Conv2DExtOrg(16, C_out, kernel_size=K, stride=S, padding=P, bias=True)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # setup loss function and optimizer
        self.loss_f = self.set_up_loss(config.loss, config.loss_weights, device=device)

        self.loss_f_test = self.set_up_loss(config.loss, config.loss_weights, device='cpu')

        self.set_up_scheduling(train_total_len=train_total_len)

        # if a load checkpoint is given, load it
        if config.load_path != None:
            self.load(config.load_path)


    def forward(self, x):
        # Pass the input to CNNT and work with the output

        pre = self.pre_cnnt(x)#.permute(0,2,1,3,4)
        #print(pre.shape)
        noise = self.cnnt(pre)
        output = noise if self.no_residual else pre - noise
        #print(output.shape)
        #output = output.permute(0,2,1,3,4)
        #print(output.shape)

        pos = self.pos_cnnt(output)

        #print(pos.shape)


        return pos.sigmoid()

    def compute_loss(self, output, targets, weights):
        # compute loss
        loss = self.loss_f(output, targets, weights)
        return loss

    def compute_loss_test(self, output, targets, weights):
        # separate loss func to compute loss on cpu on test set
        loss = self.loss_f_test(output, targets, weights)
        return loss

# -------------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    pass
