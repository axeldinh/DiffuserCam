#!/bin/sh

PSF="data/psf/psf4.png"
FILES="data/raw_data/*.png"
NITER=500

for image in $FILES
do
    echo "Processing $image with Non Negative Least-Square..."
    python3 scripts/non_neg_ls_reconstruction.py --psf_fp "$PSF" --data_fp "$image" --rg 2 --bg 1.5 --bayer --save --no_plot --n_iter "$NITER"
done;
for image in $FILES
do
    echo "Processing $image with Ridge..."
    python3 scripts/ridge_reconstruction.py --psf_fp "$PSF" --data_fp "$image" --rg 2 --bg 1.5 --bayer --save --no_plot --n_iter "$NITER"
done;
for image in $FILES
do
    echo "Processing $image with Lasso..."
    python3 scripts/lasso_reconstruction.py --psf_fp "$PSF" --data_fp "$image" --rg 2 --bg 1.5 --bayer --save --no_plot --n_iter "$NITER"
done;