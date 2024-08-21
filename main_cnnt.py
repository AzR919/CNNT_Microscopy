"""

Main file for running the CNN_Tranformer (CNNT)

Hub for the training cycle and mid-run evaluations
Uses "logging" to log data locally and "wandb" for experiment tracking

Provides an arg-parser for command line arguments
For detailed description of the arguments run:
    python3 main.py --help

    python3 main.py --h5files /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_paper/Base_All_train.h5 --test_case /isilon/lab-xue/publications/CNNT_paper/data/micro_datasets_paper/Base_Actin_test.h5 --ratio 100 0 0 --global_lr 0.0001 --num_epochs 300 --batch_size 8 --time 16 --width 128 160 --height 128 160 --loss ssim --loss_weights 1.0 --im_value_scale 0 4096 --wandb_entity nhlbi-nih --run_name backbone_training_run_0 --run_notes backbone_default_model_300_epochs

"""

import os
import time
import copy
import wandb
import logging
import argparse
import tifffile
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader

from models.LSUV import LSUVinit
from models.enhancement_model_cnnt import *
from running_inference import *
from data_utils import *
from utils import *
from models.fpn import *
from ptflops import get_model_complexity_info
from thop import profile

# -------------------------------------------------------------------------------------------------

def arg_parser():

    parser = argparse.ArgumentParser("Argument parser for CNNT")

    parser.add_argument("--project", type=str, default='CNNT', help='project name')
    parser.add_argument("--wandb_entity", type=str, default=None, help='wandb entity to link with')

    # Path arguments.
    parser.add_argument("--check_path", type=str, default='./logs/check', help='directory for saving checkpoints (model weights)')
    parser.add_argument("--model_path", type=str, default='./logs/model', help='directory for saving the final model')
    parser.add_argument("--load_path", type=str, default=None, help='path to load a specific checkpoint')
    parser.add_argument("--result_path", type=str, default='./logs/results', help='directory for saving best results')
    parser.add_argument("--log_path", type=str, default='./logs/logs', help='directory for log files')
    parser.add_argument("--test_image_path", type=str, default='./logs/images', help='directory for saving test images')

    # Data type arguments
    parser.add_argument("--h5files", nargs='+', type=str, default=[], help='path to each data file (only h5 files are accepted)')
    parser.add_argument("--val_case", nargs='+', type=str, default=None, help='special val case. (overrides normal val set)')
    parser.add_argument("--test_case", nargs='+', type=str, default=None, help='special test case. (overrides normal test set)')
    
    # Dataset arguments
    parser.add_argument('--ratio', nargs='+', type=int, default=[90,5,5], help='Ratio (as a percentage) for train/val/test divide of training data. Does allow for not using the entire dataset')
    parser.add_argument("--time", type=int, default=16, help='the max time series length of the input cutout')
    parser.add_argument('--height', nargs='+', type=int, default=[128, 160], help='height of the image patch cutout')
    parser.add_argument('--width', nargs='+', type=int, default=[128, 160], help='width of the image patch cutout')
    parser.add_argument("--num_patches_per_train_sample", type=int, default=8, help='number of patches per train sample')
    parser.add_argument("--num_patches_per_val_sample", type=int, default=8, help='number of patches per val sample')


    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=30, help='number of epochs to train for')
    parser.add_argument("--batch_size", type=int, default=8, help='size of batch to use')
    parser.add_argument("--save_cycle", type=int, default=5, help='Number of epochs between saving model')
    parser.add_argument("--num_samples_wandb", type=int, default=4, help='number of samples uploaded to wandb as video')
    parser.add_argument("--num_test_samples_wandb", type=int, default=2, help='number of test samples uploaded to wandb as video')

    # Inference arguments
    parser.add_argument("--test_only", action="store_true", help='no train or val. used to infer only')


    parser = add_shared_args(parser=parser)

    # General arguments
    parser.add_argument("--seed", type=int, default=0, help='seed for randomization')

    return parser.parse_args()

args = arg_parser()

# -------------------------------------------------------------------------------------------------

def get_seg_loss(enhanced_image, seg, seg_criterion, device='cpu'):
    
    B, T, C, H, W = enhanced_image.shape
    # segment the enhanced image
    seg_input = enhanced_image.reshape((B*T, C, H, W)).to(device)
    if seg_input.size(1) == 1:
        seg_input = seg_input.repeat(1, 3, 1, 1)
    seg_output = seg(seg_input).to(device)

    # build seg output
    target = (get_NoGT_target(seg_output)).data.to(device)

    # calculate seg. loss
    seg_loss = seg_criterion(seg_output, target)

    return seg_loss

def compute_loss(model, x, y, weights, config):

    output = model(x)

    if config.dp:
        loss = model.module.compute_loss(output, targets=y, weights=weights)
    else:
        loss = model.compute_loss(output, targets=y, weights=weights)

    return loss, output

def train(model, config, train_set, val_set, test_set, val_set_larger, test_set_larger):

    # setup
    train_loader = []
    for idx, h in enumerate(config.height):
        logging.info(f"--> train dataset {idx}, cutout shape is {train_set[idx].cutout_shape}")
        print(train_set[idx].__len__())
        train_loader.append(
            DataLoader(train_set[idx], shuffle=True, pin_memory=True, drop_last=True,
                        batch_size=config.batch_size, num_workers=12, prefetch_factor=4, #os.cpu_count()//len(config.height)
                        persistent_workers=True))

    val_dataset = DataLoader(val_set, shuffle=False, pin_memory=True, drop_last=False,
                                batch_size=config.batch_size, num_workers=12, prefetch_factor=4)

    val_dataset_larger = DataLoader(val_set_larger, shuffle=False, pin_memory=True, drop_last=False,
                                batch_size=config.batch_size, num_workers=12, prefetch_factor=4)

    print('--> done dataloader')
    
    best_model_wts = copy.deepcopy(model.state_dict())

    # TODO: DP -> DDP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #seg = fpn(2)
    
    if config.dp:
        model = nn.DataParallel(model)
        #seg = nn.DataParallel(seg)

    model.to(device)
    #seg.to(device)

    #seg_criterion = FocalLoss(gamma=2).to(device)
    
    wandb.watch(model)

    if(config.load_path is None and not config.skip_LSUV and config.num_epochs):
        t0 = time.time()
        input_data  = torch.stack([ train_set[0][i][0] for i in range(config.batch_size)])
        if(input_data.dim()==6):
            input_data = torch.reshape(input_data, (-1, *input_data.shape[2:]))
        LSUVinit(model, input_data.to(device=device), verbose=False)
        logging.info(f"LSUVinit took {time.time()-t0 : .2f} seconds ...")

    try:
        scheduler = model.module.scheduler
        scheduler_on_batch = model.module.scheduler_on_batch
    except:
        scheduler = model.scheduler
        scheduler_on_batch = model.scheduler_on_batch

    # set up meters to compute running average
    train_running_loss_meter = AverageMeter()
    #train_mse_meter = AverageMeter()
    train_L1_meter = AverageMeter()
    #train_sobel_meter = AverageMeter()
    #train_ssim_meter = AverageMeter()
    #train_ssim3D_meter = AverageMeter()
    #train_psnr_meter = AverageMeter()

    best_val_loss = np.inf

    #mse_loss_func = MSE_Complex_Loss(reduction='mean', device=device)
    l1_loss_func = Weighted_L1_Complex_Loss(reduction='mean', device=device)
    #sobel_loss_func = Weighted_Sobel_Complex_Loss(device=device)
    #ssim_loss_func = Weighted_SSIM_Complex_Loss(reduction='elementwise_mean', device=device)
    #ssim3D_loss_func = Weighted_SSIM_Complex_Loss(reduction='elementwise_mean', device=device) #Weighted_SSIM3D_Complex_Loss(reduction='elementwise_mean', device=device)
    #psnr_func = PSNR(device=device)

    epoch = 0
    for epoch in range(config.num_epochs):
        logging.info(f"---------Epoch:{epoch}/{config.num_epochs}---------")

        model.train()

        train_loader_iter = []
        for h in range(len(config.height)):
            train_loader_iter.append(iter(train_loader[h]))

        train_running_loss_meter.reset()
        #train_mse_meter.reset()
        train_L1_meter.reset()
        #train_sobel_meter.reset()
        #train_ssim_meter.reset()
        #train_ssim3D_meter.reset()
        #train_psnr_meter.reset()

        model.zero_grad()
                

        total_num_batches = len(config.height) * len(train_loader[0])
        with tqdm(total=total_num_batches) as pbar:

            indexes = np.arange(total_num_batches)
            np.random.shuffle(indexes)

            for idx in indexes:
                loader_ind = idx % len(config.height)
                x, y, key = next(train_loader_iter[loader_ind])

                x = x.to(device)
                y = y.to(device)

                weights=None

                #model.optim.zero_grad()
                loss, output = compute_loss(model, x, y, weights, config)
                #loss_seg = get_seg_loss(output, seg, seg_criterion, device=device)

                #loss = loss + 0.25 * loss_seg

                train_running_loss_meter.update(loss.item(), n=config.batch_size)

                #model.zero_grad()
                
                loss.backward()

                if(config.clip_grad_norm>0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

                model.module.optim.step() if config.dp else model.optim.step()

                #mseloss = mse_loss_func(output, y)
                #train_mse_meter.update(mseloss.item(), n=config.batch_size)

                l1_loss = l1_loss_func(output, y, weights=weights)
                #sobel_loss = sobel_loss_func(output, y, weights=weights)
                #ssim_loss = ssim_loss_func(output, y, weights=weights)
                #ssim3D_loss = ssim3D_loss_func(output, y, weights=weights)
                #psnr_value = psnr_func(output, y)

                train_L1_meter.update(l1_loss.item(), n=config.batch_size)
                #train_sobel_meter.update(sobel_loss.item(), n=config.batch_size)
                #train_ssim_meter.update(ssim_loss.item(), n=config.batch_size)
                #train_ssim3D_meter.update(ssim3D_loss.item(), n=config.batch_size)
                #train_psnr_meter.update(psnr_value.item(), n=config.batch_size)

                #wandb.log({"Running, train_loss": loss.item(), "Running, mse_loss": mseloss.item(), "Running, l1_loss": l1_loss.item(), "Running, sobel_loss": sobel_loss.item(), "Running, ssim_loss": ssim_loss.item(), "Running, ssim3D_loss": ssim3D_loss.item(), "Running, PSNR": psnr_value.item()})
                wandb.log({"Running, train_loss": loss.item(), "Running, l1_loss": l1_loss.item()})#, "Running, ssim_loss": ssim_loss.item()})

                if (scheduler is not None) and scheduler_on_batch:
                    scheduler.step()
                    curr_lr = scheduler.get_last_lr()[0]
                else:
                    curr_lr = scheduler.optimizer.param_groups[0]['lr']

                pbar.update(1)
                B, T, C, H, W = x.shape
                shape_str = f"torch.Size([{B}, {T}, {C}, {H:3.0f}, {W:3.0f}])"
                #pbar.set_description(f'Epoch {epoch}/{config.num_epochs}, tra, {shape_str}, {train_running_loss_meter.avg:.3f}, {train_mse_meter.avg:.3f}, {train_L1_meter.avg:.3f}, {train_ssim_meter.avg:.3f}, {train_ssim3D_meter.avg:.3f}, {train_sobel_meter.avg:.4f}, {train_psnr_meter.avg:.3f}, lr {curr_lr:.8f}')
                pbar.set_description(f'Epoch {epoch}/{config.num_epochs}, tra, {shape_str}, {train_running_loss_meter.avg:.3f}, {train_L1_meter.avg:.3f}, lr {curr_lr:.8f}')

        # update the loss the running mean, so we have a better view of this epoch
        epoch_train_loss = train_running_loss_meter.avg
        #epoch_train_mse_loss = train_mse_meter.avg
        epoch_train_l1_loss = train_L1_meter.avg
        #epoch_train_sobel_loss = train_sobel_meter.avg
        #epoch_train_ssim_loss = train_ssim_meter.avg
        #epoch_train_ssim3D_loss = train_ssim3D_meter.avg
        #epoch_train_psnr = train_psnr_meter.avg

        #pbar.set_postfix_str(f'Epoch {epoch}/{config.num_epochs}, tra, {epoch_train_loss:.8f}, {epoch_train_mse_loss:.4f}, {epoch_train_l1_loss:.4f}, {epoch_train_sobel_loss:.4f}, {epoch_train_ssim_loss:.4f}, {epoch_train_ssim3D_loss:.4f}, {epoch_train_psnr:.4f}')
        pbar.set_postfix_str(f'Epoch {epoch}/{config.num_epochs}, tra, {epoch_train_loss:.8f}, {epoch_train_l1_loss:.4f}')

        # Validation
        epoch_val_loss, epoch_val_mse_loss, epoch_val_l1_loss, epoch_val_sobel_loss, epoch_val_ssim_loss, epoch_val_ssim3D_loss, epoch_val_psnr = eval_validation(model,
                                                                                                                                                    epoch,
                                                                                                                                                    device,
                                                                                                                                                    val_dataset,
                                                                                                                                                    config)

        epoch_val_loss_larger, epoch_val_mse_loss_larger, epoch_val_l1_loss_larger, epoch_val_sobel_loss_larger, epoch_val_ssim_loss_larger, epoch_val_ssim3D_loss_larger, epoch_val_large_psnr = eval_validation(model,
                                                                                                                                                                                                epoch,
                                                                                                                                                                                                device,
                                                                                                                                                                                                val_dataset_larger,
                                                                                                                                                                                                config)
        # ----------------------------------------------------------------

        # log in wandb separately for hyper parameter tuning
        wandb.log({"epoch": epoch,
                   "train_loss": epoch_train_loss,
                   #"train_mse_loss": epoch_train_mse_loss,
                   "train_l1_loss": epoch_train_l1_loss,
                   #"train_sobel_loss": epoch_train_sobel_loss,
                   #"train_ssim_loss": epoch_train_ssim_loss,
                   #"train_ssim3D_loss": epoch_train_ssim3D_loss,
                   #"train_PSNR": epoch_train_psnr,

                   "val_loss": epoch_val_loss,
                   "val_mse_loss": epoch_val_mse_loss,
                   "val_l1_loss": epoch_val_l1_loss,
                   "val_sobel_loss": epoch_val_sobel_loss,
                   "val_ssim_loss": epoch_val_ssim_loss,
                   "val_ssim3D_loss": epoch_val_ssim3D_loss,
                   "val_psnr": epoch_val_psnr,

                   "val_large_FOV_loss": epoch_val_loss_larger,
                   "val_large_FOV_mse_loss": epoch_val_mse_loss_larger,
                   "val_large_FOV_l1_loss": epoch_val_l1_loss_larger,
                   "val_large_FOV_sobel_loss": epoch_val_sobel_loss_larger,
                   "val_large_FOV_ssim_loss": epoch_val_ssim_loss_larger,
                   "val_large_FOV_ssim3D_loss": epoch_val_ssim3D_loss_larger,
                   "val_large_psnr": epoch_val_large_psnr
                   }
                  )
         
        # if the scheduler should work on epoch level
        epoch_val_loss_final = (epoch_val_loss_larger+epoch_val_loss)/2 if not config.train_only else epoch_train_loss
        if (scheduler is not None) and (scheduler_on_batch == False):
            if(isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(epoch_val_loss_final)
            else:
                scheduler.step()

        # record best val loss
        if(epoch_val_loss_final<best_val_loss):
            best_val_loss = epoch_val_loss_final
            best_model_wts = copy.deepcopy(model.state_dict())

        # save the results if we find ones better than the current best
        # save_results(epoch_train_loss, epoch_val_loss_final, config, epoch)

        # save the model weights every save_cycle
        if epoch % config.save_cycle == 0:
            save_model(model, config, epoch)
    # -------------------------------------------------------
    # after the training iteration
    wandb.log({"best_val_loss": best_val_loss})
    wandb.run.summary["best_val_loss"] = best_val_loss

    # after training, do the testing
    # test at the end of training
    if config.test_case is None:
        eval_test(model, config, test_set)
        eval_test(model, config, test_set_larger)
    else:
        eval_test_image(model, config, test_set)

    if config.num_epochs: # if 0 then we're testing otherwise we do all the following
        # save the last epoch
        save_model(model, config, epoch)

        # save last model
        try:
            last_model_cpu = model.cpu().module
        except:
            last_model_cpu = model.cpu()
        model_file_name = save_cnnt_model(last_model_cpu, last_model_cpu.config, last="_last")

        wandb.save(f"{model_file_name}.pt")
        wandb.save(f"{model_file_name}.json")

        # save the best model at the end so its ready for inference
        model.load_state_dict(best_model_wts)
        try:
            best_model_cpu = model.cpu().module
        except:
            best_model_cpu = model.cpu()

        best_model_cpu.eval()
        model_file_name = save_cnnt_model(best_model_cpu, best_model_cpu.config, last="_best")

        wandb.save(f"{model_file_name}.pt")
        wandb.save(f"{model_file_name}.json")

    return best_val_loss

# -------------------------------------------------------------------------------------------------

def eval_validation(model, epoch, device, val_dataset, config):

    val_running_loss_meter = AverageMeter()
    val_mse_meter = AverageMeter()
    val_L1_meter = AverageMeter()
    val_sobel_meter = AverageMeter()
    val_ssim_meter = AverageMeter()
    val_ssim3D_meter = AverageMeter()
    val_psnr_meter = AverageMeter()

    mse_loss_func = MSE_Complex_Loss(reduction='mean', device=device)
    l1_loss_func = Weighted_L1_Complex_Loss(reduction='mean', device=device)
    sobel_loss_func = Weighted_Sobel_Complex_Loss(device=device)
    ssim_loss_func = Weighted_SSIM_Complex_Loss(device=device)
    ssim3D_loss_func = Weighted_SSIM3D_Complex_Loss(device=device)
    #ssim3D_loss_func = Weighted_MSSSIM_Complex_Loss(device=device)
    psnr_func = PSNR(device=device)

    pbar = tqdm(enumerate(val_dataset), total=len(val_dataset))

    ind = 0

    with torch.no_grad():
        model.eval()

        for idx, (x, y, key) in pbar:

            x = x.to(device)
            y = y.to(device)
            weights=None

            loss, output = compute_loss(model, x, y, weights, config)
            val_running_loss_meter.update(loss.item(), n=config.batch_size)

            ind += 1

            output = normalize_image(output, values=(0,1), clip=True)
            mseloss = mse_loss_func(output, y)
            val_mse_meter.update(mseloss.item(), n=config.batch_size)

            # weights=torch.ones(y.shape[0], device=device)
            l1_loss = l1_loss_func(output, y, weights=weights)
            sobel_loss = sobel_loss_func(output, y, weights=weights)
            ssim_loss = ssim_loss_func(output, y, weights=weights)
            ssim3D_loss = ssim3D_loss_func(output, y, weights=weights)
            psnr_val = psnr_func(output, y)

            val_L1_meter.update(l1_loss.item(), n=config.batch_size)
            val_sobel_meter.update(sobel_loss.item(), n=config.batch_size)
            val_ssim_meter.update(ssim_loss.item(), n=config.batch_size)
            val_ssim3D_meter.update(ssim3D_loss.item(), n=config.batch_size)
            val_psnr_meter.update(psnr_val.item(), n=config.batch_size)

            pbar.set_description(f'Epoch {epoch}/{config.num_epochs}, val, {x.shape}, {val_running_loss_meter.avg:.4f}, {val_mse_meter.avg:.4f}, {val_L1_meter.avg:4f}, {val_ssim_meter.avg:.4f}, {val_ssim3D_meter.avg:.4f}, {val_sobel_meter.avg:.4f}, {val_psnr_meter.avg:.4f}')

    epoch_val_loss = val_running_loss_meter.avg
    epoch_val_mse_loss = val_mse_meter.avg
    epoch_val_l1_loss = val_L1_meter.avg
    epoch_val_sobel_loss = val_sobel_meter.avg
    epoch_val_ssim_loss = val_ssim_meter.avg
    epoch_val_ssim3D_loss = val_ssim3D_meter.avg
    epoch_val_psnr = val_psnr_meter.avg

    pbar.set_postfix_str(f'Epoch {epoch}/{config.num_epochs}, val, {x.shape}, {epoch_val_loss:.4f}, {epoch_val_mse_loss:.4f}, {epoch_val_l1_loss:4f}, {epoch_val_ssim_loss:.4f}, {epoch_val_ssim3D_loss:.4f}, {epoch_val_sobel_loss:.4f}, {epoch_val_psnr:.4f}')

    if epoch % config.save_cycle == 0:
        for i in range(0, x.shape[0], config.num_samples_wandb):

            end_i = i+config.num_samples_wandb
            if(end_i>=x.shape[0]):
                end_i = x.shape[0]

            if(i>=config.num_samples_wandb):
                break

            wandb_log_vid(f"_{i}_to_{i+config.num_samples_wandb-1}_{x.shape}_{y.shape}",
                            x[i:end_i].detach().cpu().numpy(),
                            y[i:end_i].detach().cpu().numpy(),
                            output[i:end_i].detach().cpu().numpy(),
                            )

    return epoch_val_loss, epoch_val_mse_loss, epoch_val_l1_loss, epoch_val_sobel_loss, epoch_val_ssim_loss, epoch_val_ssim3D_loss, epoch_val_psnr

def eval_test(model, config, test_set):
    # setup
    test_dataset = DataLoader(test_set, shuffle=False, pin_memory=True,
                                batch_size=config["batch_size"], num_workers=8)

    # move to cuda if available to run faster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info(f"Evaluating the test set, {test_set.cutout_shape}")

    mse_loss_func = MSE_Complex_Loss(reduction='mean', device=device)
    test_loss_meter = AverageMeter()
    test_mse_loss_meter = AverageMeter()

    with torch.no_grad():
        model.eval()
        pbar = tqdm(enumerate(test_dataset), total=len(test_dataset))

        for idx, (x, y, key) in pbar:

            x = x.to(device)
            y = y.to(device)

            weights=None

            output = model(x)
            if config.dp:
                loss = model.module.compute_loss(output, targets=y, weights=weights)
            else:
                loss = model.compute_loss(output, targets=y, weights=weights)

            test_loss_meter.update(loss.item(), n=x.shape[0])
            mseloss = mse_loss_func(output, y)
            test_mse_loss_meter.update(mseloss.item(), n=config.batch_size)

    logging.info(f"test_loss_{test_set.cutout_shape}:{test_loss_meter.avg}:{test_mse_loss_meter.avg}")
    wandb.run.summary[f"test_loss_{test_set.cutout_shape}"] = test_loss_meter.avg
    wandb.run.summary[f"test_mse_loss_{test_set.cutout_shape}"] = test_mse_loss_meter.avg

    return test_loss_meter.avg

def eval_test_image(model, config, test_set):

    try:
        model_cpu = model.cpu().module
    except:
        model_cpu = model.cpu()
    # Testing on test case images provided

    model_str = generate_model_file_name(config)
    test_results_dir = os.path.join(config.test_image_path, model_str)

    test_dataset = DataLoader(test_set, batch_size=1)

    test_running_loss_meter = AverageMeter()
    test_mse_meter = AverageMeter()
    test_L1_meter = AverageMeter()
    test_sobel_meter = AverageMeter()
    test_ssim_meter = AverageMeter()
    test_ssim3D_meter = AverageMeter()
    test_psnr_meter = AverageMeter()

    mse_loss_func = MSE_Complex_Loss(reduction='mean', device='cuda')
    l1_loss_func = Weighted_L1_Complex_Loss(reduction='mean', device='cuda')
    sobel_loss_func_cpu = Weighted_Sobel_Complex_Loss(device='cpu')
    sobel_loss_func_gpu = Weighted_Sobel_Complex_Loss(device='cuda')
    ssim_loss_func_cpu = Weighted_SSIM_Complex_Loss(device='cpu')
    ssim_loss_func_gpu = Weighted_SSIM_Complex_Loss(device='cuda')
    ssim3D_loss_func_cpu = Weighted_SSIM3D_Complex_Loss(device='cpu')
    ssim3D_loss_func_gpu = Weighted_SSIM3D_Complex_Loss(device='cuda')
    psnr_func = PSNR(device='cuda')

    logging.info(f"Evaluating and loggin the custom test set")

    pbar = tqdm(enumerate(test_dataset), total=len(test_dataset))

    images_logged = 0
    cutout = (config.time, config.height[-1], config.width[-1])
    overlap = (config.time//4, config.height[-1]//4, config.width[-1]//4)

    with torch.no_grad():
        model.eval()
        model_cpu.eval()

        for idx, (x, y, key) in pbar:

            weights=None
            cuda = True
            try:
                x = x.to('cuda')
                y = y.to('cuda')
                _, output = running_inference(model, x, cutout, overlap, config.batch_size//2, "cuda")
            except:
                x = x.to('cpu')
                y = y.to('cpu')
                _, output = running_inference(model_cpu, x, cutout, overlap, config.batch_size//2, "cpu")
                cuda = False

            if cuda:
                loss = model.module.compute_loss(output, targets=y, weights=weights) if config.dp \
                    else model.compute_loss(output, targets=y, weights=weights)
            else:
                loss = model_cpu.compute_loss_test(output, targets=y, weights=weights)

            output = normalize_image(output, values=(0,1), clip=True)
            mse_loss = mse_loss_func(output, y)
            l1_loss = l1_loss_func(output, y, weights=weights)
            sobel_loss = sobel_loss_func_gpu(output, y, weights=weights) if cuda else sobel_loss_func_cpu(output, y, weights=weights)
            ssim_loss = ssim_loss_func_gpu(output, y, weights=weights) if cuda else ssim_loss_func_cpu(output, y, weights=weights)
            ssim3D_loss = ssim3D_loss_func_gpu(output, y, weights=weights) if cuda else ssim3D_loss_func_cpu(output, y, weights=weights)
            psnr_val = psnr_func(output, y)

            test_running_loss_meter.update(loss.item())
            test_mse_meter.update(mse_loss.item())
            test_L1_meter.update(l1_loss.item())
            test_sobel_meter.update(sobel_loss.item())
            test_ssim_meter.update(ssim_loss.item())
            test_ssim3D_meter.update(ssim3D_loss.item())
            test_psnr_meter.update(psnr_val.item())

            pbar.set_description(f'Test, {x.shape}, {test_running_loss_meter.avg:.4f}, {test_mse_meter.avg:.4f}, {test_L1_meter.avg:4f}, {test_ssim_meter.avg:.4f}, {test_ssim3D_meter.avg:.4f}, {test_sobel_meter.avg:.4f}, {test_psnr_meter.avg:.4f}')

            x = x[0,:,0].cpu().detach().numpy()
            y = y[0,:,0].cpu().detach().numpy()
            output = output[0,:,0].cpu().detach().numpy()

            composed_channel_wise = np.transpose(np.array([x, output, y]), (1,0,2,3))

            tifffile.imwrite(os.path.join(test_results_dir, f"{key}.tif"), composed_channel_wise, imagej=True)

            x = normalize_image(x, percentiles=(0,100))
            output = normalize_image(output, values=(np.percentile(y, 0), np.percentile(y, 100)), clip=True)
            y = normalize_image(y, percentiles=(0,100))
            T, H, W = x.shape
            composed_res = np.zeros((T, H, 3*W))
            composed_res[:,:H,0*W:1*W] = x
            composed_res[:,:H,1*W:2*W] = y
            composed_res[:,:H,2*W:3*W] = output

            temp = np.zeros_like(composed_res)
            composed_res = cv2.normalize(composed_res, temp, 0, 255, norm_type=cv2.NORM_MINMAX)

            if (images_logged + 1) > config.num_test_samples_wandb:
                continue
            images_logged += 1

            wandb.log({f"Nosiy_GT_Pred_test_image_{x.shape}_{idx}_{key}": wandb.Video(np.repeat(composed_res[:,np.newaxis], 3, axis=1).astype('uint8'), fps=8, format="gif")})

    wandb.log({"test_loss":test_running_loss_meter.avg,
               "test_mse_loss":test_mse_meter.avg,
               "test_l1_loss":test_L1_meter.avg,
               "test_sobel_loss":test_sobel_meter.avg,
               "test_ssim_loss":test_ssim_meter.avg,
               "test_ssim3D_loss":test_ssim3D_meter.avg,
               "test_psnr":test_psnr_meter.avg
    })

    wandb.run.summary[f"test_loss"] = test_running_loss_meter.avg
    wandb.run.summary[f"test_mse_loss"] = test_mse_meter.avg
    wandb.run.summary[f"test_l1_loss"] = test_L1_meter.avg
    wandb.run.summary[f"test_sobel_loss"] = test_sobel_meter.avg
    wandb.run.summary[f"test_ssim_loss"] = test_ssim_meter.avg
    wandb.run.summary[f"test_ssim3D_loss"] = test_ssim3D_meter.avg
    wandb.run.summary[f"test_psnr"] = test_psnr_meter.avg

    return test_running_loss_meter.avg

# -------------------------------------------------------------------------------------------------

def prepare_training(config, run_name=None):
    # Get current date
    now = datetime.now()
    now = now.strftime("%m-%d-%Y_T%H-%M-%S")

    config["date"] = now

    # If more than 1 gpu, then we can use DP
    # TODO: go from DP to DDP
    logging.info(f"model is trained on {torch.cuda.device_count()} gpus ...")
    #if config.test_only:
    #    config["dp"] = False
    #else:
    config["dp"] = torch.cuda.device_count() > 1

    if(config.num_samples_wandb > config.batch_size):
        config.update({"num_samples_wandb":config.batch_size}, allow_val_change=True)

    # set up directories and logs
    os.makedirs(config.check_path, exist_ok=True)
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.result_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    model_str = generate_model_file_name(config)
    test_results_dir = os.path.join(config.test_image_path, model_str)
    os.makedirs(test_results_dir, exist_ok=True)

    logging.info(f"---> run {run_name}, check_path is {config.check_path}")
    logging.info(f"---> run {run_name}, model_path is {config.model_path}")
    logging.info(f"---> run {run_name}, result_path is {config.result_path}")
    logging.info(f"---> run {run_name}, log_path is {config.log_path}")

    log_file_name = os.path.join(config.log_path, f"{now}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_name),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Configuration for this run:\n{config}")

# -------------------------------------------------------------------------------------------------

def main():

    # init wandb and config
    run = wandb.init(project=args.project, entity=args.wandb_entity, config=args, name=args.run_name, notes=args.run_notes)
    config = wandb.config
    prepare_training(config=config, run_name=run.name)

    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # load data and create model
    train_set, val_set, test_set, val_set_larger, test_set_larger = load_data(config)
    print('done loading')
    if config.test_only:
        total_steps = 0    
    else:
        total_steps = (len(train_set[0])//config.batch_size) * len(train_set) * config.num_epochs

    logging.info("Training is performed with enhanced denoising model")
    model = CNNT_enhanced_denoising_runtime(config, total_steps)

    #macs, params = get_model_complexity_info(model, (16, 1, 160, 160), as_strings=True,
    #                                       print_per_layer_stat=True, verbose=True)
    x = torch.empty(1, 16, 1, 160, 160)
    macs, params = profile(model, inputs=(x,))

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    logging.info(generate_model_file_name(model.config))
    torch.multiprocessing.set_sharing_strategy('file_system')
    if config.test_only:
        if config.dp:
            model = nn.DataParallel(model)
           
        print('--> start evaluation')
        eval_test_image(model, config, test_set)
    else:
        print('--> start training')
        train(model, config, train_set, val_set, test_set, val_set_larger, test_set_larger)

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":
    main()
