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
        # Central difference approximations
        dx = np.array([[0.5, 0, -0.5]])
        dy = np.array([[0.5], [0], [-0.5]])

        dxx = np.array([[1, -2, 1]])
        dyy = np.array([[1], [-2], [1]])

        Lx = convolve2d(inpic, dx, shape)
        Ly = convolve2d(inpic, dy, shape)
        Lxx = convolve2d(inpic, dxx, shape) 
        Lyy = convolve2d(inpic, dyy, shape)
        Lxy = convolve2d(Lx, dy, shape)

        return (Lx**2) * Lxx + 2 * Lx * Ly * Lxy + (Ly**2) * Lyy

def Lvvvtilde(inpic, shape = 'same'):
        # Central difference approximations
        dx = np.array([[0.5, 0, -0.5]])
        dy = np.array([[0.5], [0], [-0.5]])

        dxx = np.array([[1, -2, 1]])
        dyy = np.array([[1], [-2], [1]])

        Lx = convolve2d(inpic, dx, shape)
        Ly = convolve2d(inpic, dy, shape) 
        Lxx = convolve2d(inpic, dxx, shape) 
        Lyy = convolve2d(inpic, dyy, shape) 
        Lxy = convolve2d(Lx, dy, shape)

        Lxxx = convolve2d(Lxx, dx, shape)
        Lyyy = convolve2d(Lyy, dy, shape)
        Lxxy = convolve2d(Lxy, dx, shape)
        Lxyy = convolve2d(Lxy, dy, shape)

        return (Lx**3) * Lxxx + 3 * (Lx**2) * Ly * Lxxy + 3 * Lx * (Ly**2) * Lxyy + (Ly**3) * Lyyy

def extractedge(inpic, scale, threshold, shape):
        # ...
        return contours
        
def houghline(curves, magnitude, nrho, ntheta, threshold, nlines = 20, verbose = False):
        # ...
        return linepar, acc

def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
        # ...
        return linepar, acc
         

exercise = "3"


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
    tools_smoothened = gaussfft(tools, 3)

    gradmagntools = Lv(tools)
    gradmagntools_smoothened = Lv(tools_smoothened)

    hist, bins = np.histogram(gradmagntools, bins=256)
    hist_smoothened, bins_smoothened = np.histogram(gradmagntools_smoothened, bins=256)

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
    showgrey(gradmagntools_smoothened, False)
    a1.set_title("Gradient magnitude with smoothing of the image")   
    
    a2 = f.add_subplot(1, 2, 2)
    a2.bar(bins_smoothened[:-1], hist_smoothened, width=(bins_smoothened[1] - bins_smoothened[0]), color='blue', alpha=0.7)
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
    f.suptitle("Thresholded Smoothened Images with Different Thresholds", fontsize=16)
    
    for i, threshold in enumerate(thresholds):
        thresholded_image_smoothened = (gradmagntools_smoothened > threshold).astype(int)

        ax = f.add_subplot(3, 2, i + 1)
        showgrey(thresholded_image_smoothened, False)
        ax.set_title(f"Threshold = {threshold}")
    
    plt.show()


if exercise == "3":
    # # Derivative verification
    # dx = np.array([[0.5, 0, -0.5]])
    # dxx = np.array([[1, -2, 1]])
    
    # [x, y] = np.meshgrid(range(-5, 6), range(-5, 6))  # Coordinate grid
    # poly1 = x**3  # Polynomial x^3
    # poly2 = x**3 * y  # Polynomial x^3 * y

    # result1 = convolve2d(poly1, dx, mode='same')
    # result2 = convolve2d(poly1, dxx, mode='same')
    # result3 = convolve2d(result2, dx, mode='same')
    # result4 = convolve2d(poly2, dx, mode='same')
    # print("Convolution with δx on x^3: \n", result1)
    # print("Convolution with δxx on x^3: \n", result2)
    # print("Convolution with δxxx on x^3: \n", result3)
    # print("Convolution with δx on x^3 * y: \n", result4)
 
    house = np.load("Images-npy/godthem256.npy")
    scales = [0.0001, 1.0, 4.0, 16.0, 64.0]
    
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    
    ax = f.add_subplot(2, 3, 1)
    showgrey(house, False)
    ax.set_title(f"Original Image")
    for i, scale in enumerate(scales):
        ax = f.add_subplot(2, 3, i + 2)
        showgrey(contour(Lvvtilde(discgaussfft(house, scale), 'same')), False)
        ax.set_title(f"Scale = {scale}")
    
    plt.show()
    # --------------------------------------------- #
    tools = np.load("Images-npy/few256.npy")

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    
    ax = f.add_subplot(2, 3, 1)
    showgrey(tools, False)
    ax.set_title(f"Original Image")
    for i, scale in enumerate(scales):
        ax = f.add_subplot(2, 3, i + 2)
        showgrey((Lvvvtilde(discgaussfft(tools, scale), 'same')<0).astype(int), False)
        ax.set_title(f"Scale = {scale}")
        
    plt.show()


    