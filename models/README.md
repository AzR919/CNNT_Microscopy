# Directory overview: models/

Models and losses used for this project

- model.py: contains the key network layers, including CnnSelfAttention, CnnTransformer and CNNTUnet. The CnnSelfAttention impelments the CNN attention layer. CnnTransformer included CNN attention, normalization and cnn mixer, as a transformer module. CNNTUnet is a u-net design using the CNNT layer, instead of convolution layers.

- enhancement_model.py: implement image enhancement models.

- enhancement_loss.py: implement loss functions.

- load_export_model.py: used to save model at pts/onnx using a checkpoint saved weights.

- LSUV.py: initialize weights.

- pytorch_ssim: an implementation of pytorch_ssim loss.
