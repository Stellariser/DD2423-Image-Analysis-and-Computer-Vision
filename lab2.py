import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft
        
        
def deltax():
    # Sobel operator
    dxmask = np.array([[-1,  0,  1],
                       [-2,  0,  2],
                       [-1,  0,  1]]) / 8
    return dxmask

def deltay():
    # Sobel operator
    dymask = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]]) / 8
    return dymask

def Lv(inpic, shape = 'same'):
    Lx = convolve2d(inpic, deltax(), shape)
    Ly = convolve2d(inpic, deltay(), shape)
    return np.sqrt(Lx**2 + Ly**2)

def Lvvtilde(inpic, shape = 'same'):
        # ...
        return result

def Lvvvtilde(inpic, shape = 'same'):
        # ...
        return result

def extractedge(inpic, scale, threshold, shape):
        # ...
        return contours
        
def houghline(curves, magnitude, nrho, ntheta, threshold, nlines = 20, verbose = False):
        # ...
        return linepar, acc

def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
        # ...
        return linepar, acc
         

exercise = "2"


if exercise == "1":
    tools = np.load("Images-npy/few256.npy")
    dxtools = convolve2d(tools, deltax(), 'valid')
    dytools = convolve2d(tools, deltay(), 'valid')

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f.add_subplot(1, 3, 1)
    showgrey(tools, False)
    a1.set_title("Original figure")

    a2 = f.add_subplot(1, 3, 2)
    showgrey(dxtools, False)
    a2.set_title("Derivative in the x direction")

    a3 = f.add_subplot(1, 3, 3)
    showgrey(dytools, False)
    a3.set_title("Derivative in the y direction")

    plt.show()


if exercise == "2":
    tools = np.load("Images-npy/few256.npy")
    tools_smoothed = gaussfft(tools, 3)

    gradmagntools = Lv(tools)
    gradmagntools_smoothed = Lv(tools_smoothed)

    hist, bins = np.histogram(gradmagntools, bins=256)
    hist_smoothed, bins_smoothed = np.histogram(gradmagntools_smoothed, bins=256)

    # --------------------------------------------- #
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f.add_subplot(1, 2, 1)
    showgrey(gradmagntools, False)
    a1.set_title("Gradient magnitude")   
    
    a2 = f.add_subplot(1, 2, 2)
    a2.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='blue', alpha=0.7)
    a2.set_title("Histogram of Gradient Magnitude")
    a2.set_xlabel("Gradient Magnitude")
    a2.set_ylabel("Frequency")

    plt.show()
    # --------------------------------------------- #
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f.add_subplot(1, 2, 1)
    showgrey(gradmagntools_smoothed, False)
    a1.set_title("Gradient magnitude with smoothing of the image")   
    
    a2 = f.add_subplot(1, 2, 2)
    a2.bar(bins_smoothed[:-1], hist_smoothed, width=(bins_smoothed[1] - bins_smoothed[0]), color='blue', alpha=0.7)
    a2.set_title("Histogram of Gradient Magnitude with smoothing of the image")
    a2.set_xlabel("Gradient Magnitude")
    a2.set_ylabel("Frequency")

    plt.show()
    # --------------------------------------------- #
    thresholds = [10, 15, 20, 25, 30, 35]

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f.suptitle("Thresholded Images with Different Thresholds", fontsize=16)
    
    for i, threshold in enumerate(thresholds):
        thresholded_image = (gradmagntools > threshold).astype(int)

        ax = f.add_subplot(3, 2, i + 1)
        showgrey(thresholded_image, False)
        ax.set_title(f"Threshold = {threshold}")
    
    plt.show()
    # --------------------------------------------- #
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f.suptitle("Thresholded Smoothed Images with Different Thresholds", fontsize=16)
    
    for i, threshold in enumerate(thresholds):
        thresholded_image_smoothed = (gradmagntools_smoothed > threshold).astype(int)

        ax = f.add_subplot(3, 2, i + 1)
        showgrey(thresholded_image_smoothed, False)
        ax.set_title(f"Threshold = {threshold}")
    
    plt.show()

    