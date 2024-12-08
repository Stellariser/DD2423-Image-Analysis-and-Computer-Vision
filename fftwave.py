import numpy as np
import matplotlib.pyplot as plt
from Functions import showgrey

def fftwave(u, v, sz = 128):
	Fhat = np.zeros([sz, sz])
	Fhat[u, v] = 1
	
	F = np.fft.ifft2(Fhat)
	Fabsmax = np.max(np.abs(F))

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)
	a1 = f.add_subplot(3, 2, 1)
	showgrey(Fhat, False)
	a1.title.set_text("Fhat: (u, v) = (%d, %d)" % (u, v))
	
	# What is done by these instructions?
	# This block of code calculates “wrapped coordinates” (uc, vc) for the indices (u, v) in the Fourier domain. 
	# The transformation essentially re-centers the frequency coordinates for more intuitive visualization, adjusting for the symmetry in DFT. 
	# It uses the fact that low frequencies are centered, while high frequencies are shifted to the edges. 
	# For indices greater than half the size, the code subtracts the size to reflect negative frequencies, so they range from -sz/2 to sz/2.
	if u < sz/2:
		uc = u
	else:
		uc = u - sz
	if v < sz/2:
		vc = v
	else:
		vc = v - sz

	wavelength = 1 / np.sqrt(uc**2 + vc**2) if uc != 0 or vc != 0 else 0
	amplitude  = 1.0 / sz * sz
	
	a2 = f.add_subplot(3, 2, 2)
	showgrey(np.fft.fftshift(Fhat), False)  # Shift the zero-frequency component to the center of the spectrum
	a2.title.set_text("centered Fhat: (uc, vc) = (%d, %d)" % (uc, vc))
	
	a3 = f.add_subplot(3, 2, 3)
	showgrey(np.real(F), False, 64, -Fabsmax, Fabsmax)
	a3.title.set_text("real(F)")
	
	a4 = f.add_subplot(3, 2, 4)
	showgrey(np.imag(F), False, 64, -Fabsmax, Fabsmax)
	a4.title.set_text("imag(F)")
	
	a5 = f.add_subplot(3, 2, 5)
	showgrey(np.abs(F), False, 64, -Fabsmax, Fabsmax)
	a5.title.set_text("abs(F) (amplitude %f)" % amplitude)
	
	a6 = f.add_subplot(3, 2, 6)
	showgrey(np.angle(F), False, 64, -np.pi, np.pi)
	a6.title.set_text("angle(F) (wavelength %f)" % wavelength)
	
	plt.show()

