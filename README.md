
# CNNT Microscopy

Implementation of Convolutional Neural Network Transformer (CNNT) for microscopy.

A novel architecture that, together with backbone fine-tuning training scheme, pushes the State-Of-The-Art in microscopy image denoising and enhancement.

For details, the pre-print paper is present here: https://arxiv.org/abs/2404.04726

## Convolutional Neural Network Transformer

We propose a new network architecture for dynamic image processing. This architecture includes a novel network layer, the convolutional neural network transformer (CNNT), which takes the input tensor [B, T, C_in, H, W] and produces the output tensor [B, T, C_out, H, W]. 

The key innovation is to design and implement a CNN attention layer. Compared to the conventional attention layer utilizing the linear layers, the CNN attention can take in large images with high spatial matrix size and avoids the explosion of number of parameters.

## Repository overview

The repo is organized as the following:
- main.py: main training interface, implementing the call to CNNT U-net.
- microscopy_dataset.py: pytorch dataset implementation for custom microscopy datasets.
- running_inference.py: helper for running model on complete images.
- data_utils.py: defines load_data functions.
- utils.py: general helpers.
- models/: contains the models and losses, including the implementation of CNNT.

## Loss functions

Main loss functions (found in models/enhancement_losses.py) used are:
- L2 norm : `Weighted_MSE_Complex_Loss`, L2 norm
- L1 norm : `Weighted_L1_Complex_Loss`, L1 norm 
- Sobel norm : `Weighted_Sobel_Complex_Loss`, a loss focusing on the edges of output and target images
- SSIM loss : `Weighted_SSIM_Complex_Loss`, the SSIM metric to measure how close two images are

More than one loss can be combined into `Image_Enhancement_Combined_Loss`. User can supply the weights to balance multiple losses, as `--loss mse l1 sobel ssim --loss_weights 1.0 10.0 100.0 100.0`

## Multi-patch training

It is found that the training will be better regularized if patches with differnet sizes are supplied. A `multi-patch` training strategy is developed in the repo by defining multiple patches sizes:

```
parser.add_argument('--height', nargs='+', type=int, default=[64, 96, 128, 160], help='height of the image patch cutout')

parser.add_argument('--width', nargs='+', type=int, default=[64, 96, 128, 160], help='width of the image patch cutout')
```

This will instantiate four data loaders. Each will create a patch size (e.g. [64,64], [96,96], [128,128], [160,160]). In training, every data loader will be called interleaved.

To use multi-patch training mode, the `norm_mode` should be `batch` or `instance`, because the `layer` normalization cannot handle inputs with different matrix sizes.

## Usage of the repo

### Requirement

- User should have a wandb account and log in with `wandb login` to log and track metrics and images. (It can be disabled with `wandb disabled` if the user wishes to do so.)

### Create a python3 virtual environment

Create and setup a virtual environment using `requirements.txt`
```
mkdir ./cnnt_venv
python3 -m venv ./cnnt_venv/
source ./cnnt_venv/bin/activate
pip install -r requirements.txt
```

### Run the backbone training

```
python3 main.py \
--h5files <path_to_first_train_h5_file> <path_to_second_train_h5file> <...> \
--test_case <path_to_test_h5_file> \
--ratio 100 0 0 \
--global_lr 0.0001 \
--num_epochs 300 --batch_size 8 \
--time 16 --width 128 160 --height 128 160 \
--loss ssim --loss_weights 1.0 \
--wandb_entity <your_wandb_account> \
--run_name backbone_training_run_0 --run_notes backbone_default_model_300_epochs
```

### Run the fine-tuning

```
python3 main.py \
--h5files <path_to_first_train_h5_file> <path_to_second_train_h5file> <...> \
--fine_samples 10 \
--test_case <path_to_test_h5_file> \
--global_lr 0.000025 --skip_LSUV \
--num_epochs 30 --batch_size 8 \
--time 16 --width 128 160 --height 128 160 \
--loss ssim mse --loss_weights 1.0 0.1 \
--wandb_entity <your_wandb_account> \
--run_name finetuning_run_0 --run_notes finetuning_default_model_with_30_epochs \
--load_path <path_to_.pt_saved_model_file>
```
