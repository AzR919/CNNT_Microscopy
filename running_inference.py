"""
File for just inference given model full image and
desired cutout and overlap
"""
import torch
import numpy as np
from skimage.util.shape import view_as_windows


def running_inference(model, image, 
                    cutout=(24,256,256), overlap=(4,64,64), 
                    batch_size=8, device="cpu"):
    """
    runs inference by breaking image into cutout shapes with overlaps
    running the patches through the model and then repatching
    @args:
        model: the model to run inference with
        image: the image to run inference on
                requires the image to have ndim==3 or ndim==5
                ndim==5 requires to be squeezable to ndim==3
                takes either numpy or torch.tensor array
                (going down to ndim==3 makes coding easier and cleaner)
        cutout: the patch shape for each cutout
                requires len(cutout)==3
        overlap: the number of pixels to overlap
                requires len(overlap)==3
        batch_size: the batch_size for model input
        device: the device to run inference on
    """
    #-----------------------------------------------------------------------
    # setup the model and image
    model = model.to(device)
    model.eval()

    if image.ndim == 5:
        B, T, C, H, W = image.shape
        assert B==1 and C==1
    try:
        image = image.cpu().detach().numpy().squeeze()
    except:
        image = image.squeeze()
    #-----------------------------------------------------------------------
    # some constant used several times
    d_type = image.dtype

    TO, HO, WO = image.shape            # original
    Tc, Hc, Wc = cutout                 # cutout
    To, Ho, Wo = overlap                # overlap
    Ts, Hs, Ws = Tc-To, Hc-Ho, Wc-Wo    # sliding window shape
    #-----------------------------------------------------------------------
    # padding the image so we have a complete coverup 
    # in each dim we pad the left side by overlap
    # and then cover the right side by what remains from the sliding window
    image_pad = np.pad(image, 
                        ((To, Ts-TO%Ts),
                        (Ho, Hs-HO%Hs),
                        (Wo, Ws-WO%Ws)),
                        "symmetric")
    #-----------------------------------------------------------------------
    # breaking the image down into patches
    # and remembering the length in each dimension
    image_patches = view_as_windows(image_pad, cutout, (Ts, Hs, Ws))
    NT, NR, NC, _, _, _ = image_patches.shape

    image_batch = image_patches.reshape(-1, *cutout) # shape:(B,T,H,W)
    #-----------------------------------------------------------------------
    # inferring each patch in length of batch_size
    image_batch_pred = np.zeros_like(image_batch, dtype=d_type)

    for i in range(0, image_batch.shape[0], batch_size):

        x_in = image_batch[i:i+batch_size]
        x_in = torch.from_numpy(x_in[:,:,np.newaxis]).to(device)
        # conver to torch and back to numpy
        image_batch_pred[i:i+batch_size] = model(x_in).cpu().detach().numpy().squeeze()
    #-----------------------------------------------------------------------
    # setting up the weight matrix
    # matrix_weight defines how much a patch contributes to a pixel
    # image_wgt is the sum of all weights. easier calculation for result
    matrix_weight = np.ones((cutout), dtype=d_type)

    for t in range(To):
        matrix_weight[t] *= ((t+1)/To)
        matrix_weight[-t-1] *= ((t+1)/To)
    
    for h in range(Ho):
        matrix_weight[:,h] *= ((h+1)/Ho)
        matrix_weight[:,-h-1] *= ((h+1)/Ho)

    for w in range(Wo):
        matrix_weight[:,:,w] *= ((w+1)/Wo)
        matrix_weight[:,:,-w-1] *= ((w+1)/Wo)

    image_wgt = np.zeros_like(image_pad, dtype=d_type) # filled in the loop below
    matrix_rep = np.repeat(matrix_weight[np.newaxis,:,:], NT*NR*NC, axis=0).reshape(*image_patches.shape)
    #-----------------------------------------------------------------------
    # Putting the patches back together
    # image_test and image_tmp used to make sure reshape worked correctly
    image_batch_pred = image_batch_pred.reshape(*image_patches.shape)
    # image_test = image_batch.reshape(*image_patches.shape)

    # image_tmp = np.zeros_like(image_pad, dtype=d_type)
    image_prd = np.zeros_like(image_pad, dtype=d_type)

    for nt in range(NT):
        for nr in range(NR):
            for nc in range(NC):
                image_wgt[Ts*nt:Ts*nt+Tc, Hs*nr:Hs*nr+Hc, Ws*nc:Ws*nc+Wc] += matrix_rep[nt, nr, nc]
                # image_tmp[Ts*nt:Ts*nt+Tc, Hs*nr:Hs*nr+Hc, Ws*nc:Ws*nc+Wc] += matrix_weight * image_test[nt, nr, nc]
                image_prd[Ts*nt:Ts*nt+Tc, Hs*nr:Hs*nr+Hc, Ws*nc:Ws*nc+Wc] += matrix_weight * image_batch_pred[nt, nr, nc]

    # image_tmp /= image_wgt
    image_prd /= image_wgt
    #-----------------------------------------------------------------------
    # remove the extra padding
    # np.testing.assert_allclose(image_tmp, image_pad, rtol=1e-4)
    # image_tmp = image_tmp[To:To+TO, Ho:Ho+HO, Wo:Wo+WO]
    # np.testing.assert_allclose(image_tmp, image, rtol=1e-4)
    image_fin = image_prd[To:To+TO, Ho:Ho+HO, Wo:Wo+WO]
    #-----------------------------------------------------------------------
    # return a 3dim numpy and 5dim torch.tensor for easier followups
    return image_fin, torch.from_numpy(image_fin[np.newaxis,:,np.newaxis]).to(device)
