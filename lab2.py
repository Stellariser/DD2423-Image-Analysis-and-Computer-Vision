import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft

# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

# Exercise 1
if 0:
	print("That was a stupid idea")
        
        
def deltax():
        # ....
        return dxmask

def deltay():
        # ....
        return dymask

def Lv(inpic, shape = 'same'):
        # ...
        return result

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
         
