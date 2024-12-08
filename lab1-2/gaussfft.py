import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def dnorm(x, mu, std):
    return 1 / (2 * np.pi * std) * np.exp(-np.power(x - mu, 2) / (2 * std))


def gaussfft(pic, t):
    row, col = np.shape(pic)

    kernel_1D = np.linspace(-(row//2), row//2, row)
    
    for i in range(row):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, t)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    Phat = fft2(pic)
    Ghat = fft2(fftshift(kernel_2D))
    result = np.real(ifft2(Phat * Ghat))

    return result
