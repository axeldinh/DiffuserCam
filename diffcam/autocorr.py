import numpy as np


def autocorr2d(vals, pad_mode="reflect"):
    """
    Compute 2-D autocorrelation of image via the FFT.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Return
    ------
    autocorr : py:class:`~numpy.ndarray`
    """
    
    n, m = vals.shape
    vals = np.pad(vals, ((n//2, n//2), (m//2,m//2)), mode = pad_mode)
    fft = np.fft.rfft2(vals)
    auto_corr = np.fft.fftshift(np.fft.irfft2(abs(fft)**2))

    return auto_corr[n//2:-n//2, m//2:-m//2]
