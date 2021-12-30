"""
This computes the reconstructions of our mini dataset and evaluates them against the original data.
This is done using the lasso algorithm.

To use it either use:
```bash
python scripts/lasso_myDataset.py --data myDataset --save
```
to compute the reconstructions and evaluate them, or:
```bash
python scripts/lasso_myDataset.py --data myDataset --load
```
to evaluate pre-computed reconstructions.
"""

import glob
import os
import click
import pathlib as plib
import time
import matplotlib.pyplot as plt

import numpy as np
from pycsou.func.base import ProxFuncHStack
from diffcam.io import load_data, load_image
from diffcam.plot import plot_image
from diffcam.metric import *
from pycsou.opt.proxalgs import APGD
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.linop.conv import Convolve2D
from pycsou.func import DiffFuncHStack

@click.command()
@click.option(
    "--load",
    is_flag=True,
    help="If True, the reconstruction will be loaded for the evaluation (no reconstruction)."
)
@click.option(
    "--data",
    type=str,
    help="Dataset to work on.",
)
@click.option(
    "--n_iter",
    type=int,
    default=500,
    help="Number of iterations.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save reconstructions."
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
def lasso_myDataset(data, n_iter, gray, save, load):
    
    assert data is not None
    
    original_dir = os.path.join(data, "original")
    raw_data_dir = os.path.join(data, "raw_data")
    psf_fp = os.path.join(data, "psf.png")
    downsample = 4

    # determine files
    files = glob.glob(raw_data_dir + "/*")
    files = [os.path.basename(fn) for fn in files]
    print("Number of files : ", len(files))
        
    save_path = "results_lasso_myDataset"
    save_path = plib.Path(__file__).parent / save_path
    if not save_path.exists() and save:
        save_path.mkdir(exist_ok=False)
        
    print("\nLooping through files...")
    cpu_times = []
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    for fn in files:
        
        bn = os.path.basename(fn).split(".")[0]
        print(f"\n{bn}")
        
        if not load:
            
            # Load the data + psf
            psf, data = load_data(
                psf_fp=psf_fp,
                data_fp=os.path.join(raw_data_dir, fn),
                downsample=downsample,
                bayer=True,
                blue_gain=1.5,
                red_gain=2.,
                plot=True,
                flip=False,
                gamma=None,
                gray=gray,
                single_psf=False,
            )
            
            # Setup the reconstruction
            start_time = time.time()
            if gray:
                H = Convolve2D(size = psf.size, filter = psf, shape = psf.shape)
                H.compute_lipschitz_cst()

                l22_loss = 0.5 * SquaredL2Loss(dim = H.shape[0], data = data.flatten())
                F = l22_loss * H
                lambda_ = 1e-1
                G  = lambda_ * L1Norm(dim = data.size)
            
            else:
                # Apply the reconstruction on each channel
        
                H1 = Convolve2D(size = data[:,:,0].size, filter = psf[:,:,0], shape = data[:,:,0].shape)
                H1.compute_lipschitz_cst()
                loss1 = 0.5 * SquaredL2Loss(dim = H1.shape[0], data = data[:,:,0].flatten()) * H1
                
                H2 = Convolve2D(size = data[:,:,1].size, filter = psf[:,:,1], shape = data[:,:,1].shape)
                H2.compute_lipschitz_cst()
                loss2 = 0.5 * SquaredL2Loss(dim = H2.shape[0], data = data[:,:,1].flatten()) * H2
                
                H3 = Convolve2D(size = data[:,:,2].size, filter = psf[:,:,2], shape = data[:,:,2].shape)
                H3.compute_lipschitz_cst()
                loss3 = 0.5 * SquaredL2Loss(dim = H3.shape[0], data = data[:,:,2].flatten()) * H3
                
                F = DiffFuncHStack(loss1, loss2, loss3)
                lambda_ = 1e-1
                G = lambda_ * ProxFuncHStack(L1Norm(dim = data[:,:,0].size), L1Norm(dim = data[:,:,1].size), L1Norm(dim = data[:,:,2].size))

            print(f"setup time : {time.time() - start_time} s")
            
            # Perform the reconstruction

            start_time = time.time()
            apgd = APGD(dim = data.size, F=F, acceleration='CD', max_iter = n_iter, accuracy_threshold=1e-5)
            estimate, _, _ = apgd.iterate()
            cpu_times.append(time.time() - start_time)
            print(f"proc time : {cpu_times[-1]} s")
        
            if gray:
                result = estimate['iterand'].reshape(data.shape)
            else:
                result = estimate['iterand']
                n, m = data[:,:,0].shape
                result = result.reshape((3*n, m))
                result = np.concatenate(
                    [result[:n, :, None], result[n:2*n, :,None], result[2*n:, :, None]], axis = -1)
            result = (result-np.min(result)) / (np.max(result) - np.min(result))
            if save:
                output_fn = os.path.join(save_path, f"{bn}.png")
                ax = plot_image(result, gamma = None) # Wrong but too late
                plt.savefig(str(save_path) + "/" + fn, bbox_inches= "tight", format="png")
                plt.clf()
                print(f"Files saved to : {save_path}")
                
        else:
            print("Results directly loaded from pre-computed reconstructions.")
        
        result_fn = os.path.join(save_path, f"{bn}.png")
        result = load_image(result_fn)
        true_fn = glob.glob(original_dir + "/" + str(bn).split(".")[0]+"*")[0]
        true = load_image(true_fn)
        print("\nComputing the metrics...")
        
        #### PUT CODE FOR METRICS HERE
        # crop(result, filename)
        # mse_scores(mse(true, result))
        # psnr_scores(psnr(true, result))
            

    if save:
        print(f"\nReconstructions saved to : {save_path}")
    
    print("\nCPU time (avg)", np.mean(cpu_times))
    print("MSE (avg)", np.mean(mse_scores))
    print("PSNR (avg)", np.mean(psnr_scores))
    print("SSIM (avg)", np.mean(ssim_scores))
    print("LPIPS (avg)", np.mean(lpips_scores))
                
if __name__ == "__main__":
    lasso_myDataset()