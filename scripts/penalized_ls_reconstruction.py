"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.

```bash
python scripts/reconstruction_template.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png
```

"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from diffcam.io import load_data
from diffcam.plot import plot_image
from pycsou.opt.proxalgs import PDS
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm, NonNegativeOrthant
from pycsou.linop.diff import Gradient
from pycsou.linop.conv import Convolve2D
from pycsou.func import DiffFuncHStack, ProxFuncHStack

@click.command()
@click.option(
    "--psf_fp",
    type=str,
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=str,
    help="File name for raw measurement data.",
)
@click.option(
    "--n_iter",
    type=int,
    default=500,
    help="Number of iterations.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--disp",
    default=50,
    type=int,
    help="How many iterations to wait for intermediate plot/results. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Whether to flip image.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="Whether to no plot.",
)
@click.option(
    "--bg",
    type=float,
    help="Blue gain.",
)
@click.option(
    "--rg",
    type=float,
    help="Red gain.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
def reconstruction(
    psf_fp,
    data_fp,
    n_iter,
    downsample,
    disp,
    flip,
    gray,
    bayer,
    bg,
    rg,
    gamma,
    save,
    no_plot,
    single_psf,
):
    psf, data = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=not no_plot,
        flip=flip,
        gamma=gamma,
        gray=gray,
        single_psf=single_psf,
    )

    if disp < 0:
        disp = None
    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "PENALIZED_LS_RECONSTRUCTION_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    # TODO : setup for your reconstruction algorithm
    #n, m = psf.shape
    if gray:
        
        conv = Convolve2D(size = psf.size, filter = psf, shape = psf.shape)
        conv.compute_lipschitz_cst()

        l22_loss = 0.5 * SquaredL2Loss(dim = conv.shape[0], data = data.flatten())
        F = l22_loss * conv
                
        D = Gradient(shape = data.shape)
        D.compute_lipschitz_cst()
        lambda_ = 1e-1
        H = lambda_ * L1Norm(dim = D.shape[0])
        G = NonNegativeOrthant(dim = data.size)
        
        
    
    else:
        
        conv1 = Convolve2D(size = data[:,:,0].size, filter = psf[:,:,0], shape = data[:,:,0].shape)
        conv1.compute_lipschitz_cst()
        loss1 = 0.5 * SquaredL2Loss(dim = conv1.shape[0], data = data[:,:,0].flatten()) * conv1
        
        conv2 = Convolve2D(size = data[:,:,1].size, filter = psf[:,:,1], shape = data[:,:,1].shape)
        conv2.compute_lipschitz_cst()
        loss2 = 0.5 * SquaredL2Loss(dim = conv2.shape[0], data = data[:,:,1].flatten()) * conv2
        
        conv3 = Convolve2D(size = data[:,:,2].size, filter = psf[:,:,2], shape = data[:,:,2].shape)
        conv3.compute_lipschitz_cst()
        loss3 = 0.5 * SquaredL2Loss(dim = conv3.shape[0], data = data[:,:,2].flatten()) * conv3
        
        F = DiffFuncHStack(loss1, loss2, loss3)
        
        D = Gradient(shape = data.shape)
        D.compute_lipschitz_cst()
        lambda_ = 1e-1
        H = lambda_ * L1Norm(dim=D.shape[0])
        G = NonNegativeOrthant(dim = data.size)
                                            
        
        
    
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    # TODO : apply your reconstruction
    
    pds = PDS(dim = data.size, F=F, G=G, H=H, K=D, max_iter = n_iter, accuracy_threshold  = 1e-5)
    estimate, _, _ = pds.iterate()
    print(f"proc time : {time.time() - start_time} s")
    
    if gray:
        result = estimate['primal_variable'].reshape(data.shape)
    else:
        result = estimate['primal_variable']
        n, m = data[:,:,0].shape
        result = result.reshape((3*n, m))
        result = np.concatenate(
            [result[:n, :, None], result[n:2*n, :,None], result[2*n:, :, None]], axis = -1)

    if not no_plot:
        ax = plot_image(result, gamma = gamma)
        plt.show()
        plt.clf()
    if save:
        ax = plot_image(result, gamma = gamma)
        plt.savefig(str(save) + "/" + data_fp.split("/")[-1], bbox_inches= "tight", format="png")
        plt.clf()
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    reconstruction()
