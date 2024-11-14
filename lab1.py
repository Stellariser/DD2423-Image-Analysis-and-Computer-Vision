import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from scipy.ndimage import rotate
from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


exercise = "1.3"


if exercise == "1.3":
	coordinates = [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]
	for u, v in coordinates:
		fftwave(u, v)

	# ---------------------------------------------------------------------- #

	Fhat = np.zeros([128, 128])
	Fhat[-3, 1] = 1
	
	# Perform inverse FFT with and without fftshift
	F = np.fft.ifft2(np.fft.fftshift(Fhat))  # Inverse FFT with wrapping
	F_wrapped = np.fft.ifft2(Fhat)  # Inverse FFT without wrapping
	
	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(2, 1, 1)
	showgrey(np.real(F), False)
	a1.title.set_text("F without wrapped coordinates")
	

	a2 = f.add_subplot(2, 1, 2)
	showgrey(np.real(F_wrapped), False) 
	a2.title.set_text("F with wrapped coordinates")

	plt.show()
	


if exercise == "1.4":
	F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
	G = F.T
	H = F + 2 * G

	Fhat = fft2(F)
	Ghat = fft2(G)
	Hhat = fft2(H)
	
	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(3, 3, 1)
	showgrey(F, False)
	a1.title.set_text("Image F")
	
	a2 = f.add_subplot(3, 3, 2)
	showgrey(G, False)
	a2.title.set_text("Image G")
	
	a3 = f.add_subplot(3, 3, 3)
	showgrey(H, False)
	a3.title.set_text("Image H")
	
	a4 = f.add_subplot(3, 3, 4)
	showgrey(np.log(1 + np.abs(Fhat)), False)
	a4.title.set_text("Fourier Spectrum of F")
	
	a5 = f.add_subplot(3, 3, 5)
	showgrey(np.log(1 + np.abs(Ghat)), False)
	a5.title.set_text("Fourier Spectrum of G")
	
	a6 = f.add_subplot(3, 3, 6)
	showgrey(np.log(1 + np.abs(Hhat)), False)
	a6.title.set_text("Fourier Spectrum of H")

	a7 = f.add_subplot(3, 3, 7)
	showgrey(np.log(1 + np.abs(fftshift(Fhat))), False)
	a7.title.set_text("Centered Fourier Spectrum of F")

	a8 = f.add_subplot(3, 3, 8)
	showgrey(np.log(1 + np.abs(fftshift(Ghat))), False)
	a8.title.set_text("Centered Fourier Spectrum of G")

	a9 = f.add_subplot(3, 3, 9)
	showgrey(np.log(1 + np.abs(fftshift(Hhat))), False)
	a9.title.set_text("Centered Fourier Spectrum of H")
	
	plt.show()


if exercise == "1.5":
	F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
	G = F.T

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(1, 3, 1)
	showgrey(F * G, False)
	a1.title.set_text("F * G")

	a2 = f.add_subplot(1, 3, 2)
	showfs(np.fft.fft2(F * G), False)
	a2.title.set_text("fft(F * G)")

	a3 = f.add_subplot(1, 3, 3)
	Fhat = fft2(F)
	Ghat = fft2(G)
	showgrey(np.log(1 + np.abs(convolve2d(Fhat, Ghat, mode='same', boundary='wrap') / (128 ** 2))), False)
	a3.title.set_text("conv(Fhat, Ghat)")

	plt.show()


if exercise == "1.6":
	F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
		np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
	

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(1, 2, 1)
	showgrey(F, False)
	a1.title.set_text("Scaled Image F")

	a2 = f.add_subplot(1, 2, 2)
	Fhat = np.fft.fft2(F)
	showfs(Fhat, False)
	a2.title.set_text("Fourier Spectrum of Scaled Image F")

	plt.show()


if exercise == "1.7":
	angles = [0, 30, 45, 60, 90]
	F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
		np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)

	fig, axs = plt.subplots(len(angles), 6, figsize=(24, 4 * len(angles)))
	fig.subplots_adjust(hspace=0.4, wspace=0.4)

	for i, alpha in enumerate(angles):
		# Rotate the image and compute its Fourier spectrum
		G = rotate(F, angle=alpha, reshape=False)
		Ghat = fft2(G)
		Hhat = rotate(fftshift(Ghat), angle=-alpha, reshape=False)

		# Display the rotated image
		axs[i, 0].imshow(G, cmap='gray')
		axs[i, 0].set_title(f"Rotated Image (alpha={alpha}°)", fontsize=7)
		axs[i, 0].axis('off')

		# Display the Fourier spectrum of the rotated image
		axs[i, 1].imshow(np.log(1 + np.abs(fftshift(Ghat))), cmap='gray')
		axs[i, 1].set_title(f"Fourier Spectrum of Rotated Image (alpha={alpha}°)", fontsize=7)
		axs[i, 1].axis('off')

		# Display the rotated spectrum back to the original orientation
		axs[i, 2].imshow(np.log(1 + np.abs(Hhat)), cmap='gray')
		axs[i, 2].set_title(f"Rotated Spectrum Back (alpha={-alpha}°)", fontsize=7)
		axs[i, 2].axis('off')

		# Display the original Fourier spectrum for reference
		Fhat = fft2(F)
		axs[i, 3].imshow(np.log(1 + np.abs(fftshift(Fhat))), cmap='gray')
		axs[i, 3].set_title("Original Fourier Spectrum", fontsize=7)
		axs[i, 3].axis('off')

		# Display the magnitude of the Fourier spectrum of the rotated image
		axs[i, 4].imshow(np.abs(fftshift(Ghat)), cmap='gray')
		axs[i, 4].set_title(f"Magnitude (alpha={alpha}°)", fontsize=7)
		axs[i, 4].axis('off')

		# Display the phase of the Fourier spectrum of the rotated image
		axs[i, 5].imshow(np.angle(fftshift(Ghat)), cmap='gray')
		axs[i, 5].set_title(f"Phase (alpha={alpha}°)", fontsize=7)
		axs[i, 5].axis('off')

	plt.show()


if exercise == "1.8":
	from Functions import pow2image, randphaseimage
	import os
	
	image_paths = [
        "Images-npy/phonecalc128.npy",
        "Images-npy/few128.npy",
        "Images-npy/nallo128.npy"
    ]
	a = 1e-3

	for img_path in image_paths:
		if os.path.exists(img_path):
			img = np.load(img_path)

			# 替换功率谱后的图像
			# Power spectrum as negative power of two
			img_pow2 = pow2image(img, a)

			# 随机化相位后的图像
			img_randphase = randphaseimage(img)

			f = plt.figure()
			f.subplots_adjust(wspace=0.2, hspace=0.4)
			plt.rc('axes', titlesize=10)

			a1 = f.add_subplot(1, 3, 1)
			showgrey(img, False)
			a1.title.set_text("Original Image")

			a2 = f.add_subplot(1, 3, 2)
			showgrey(img_pow2, False)
			a2.title.set_text("Image with replaced power spectrum")

			a3 = f.add_subplot(1, 3, 3)
			showgrey(img_randphase, False)
			a3.title.set_text("Image with random phase")

			plt.show()

		else:
			print(f"Image {img_path} not found.")


if exercise == "2.3":
	from Functions import deltafcn,variance
	from gaussfft import gaussfft

	"""执行高斯卷积分析，包括脉冲响应和图像平滑效果展示。"""

	# Different parameters for test
	t_values = {
		"variance_test": [0.1, 0.3, 1.0, 10.0, 100.0],
		"blur_test": [1.0, 4.0, 16.0, 64.0, 256.0]
	}
	img = np.load("Images-npy/genevepark128.npy")

	# Analyzing impulse respons and covariances
	fig1, axs1 = plt.subplots(1, 5, figsize=(15, 3))
	fig1.subplots_adjust(wspace=0.3)
	plt.rc('axes', titlesize=10)

	for i, t in enumerate(t_values["variance_test"]):
		psf = gaussfft(deltafcn(128, 128), t)
		var = variance(psf)
		var = [[round(j, 3) for j in var[i]] for i in range(len(var))]

		print(f"Variance for t={t}: {var:}")

		# Impulse respons visualization
		ax_img = axs1[i]
		ax_img.imshow(psf, cmap='gray')
		ax_img.set_title(f'Impulse respons visualization:\nt={t}\nvar={var:}')
		ax_img.axis('off')
	

	# Analyzing image bluring effect
	fig2, axs2 = plt.subplots(1, 5, figsize=(15, 3))
	fig2.subplots_adjust(wspace=0.3)
	plt.rc('axes', titlesize=10)

	for i, t in enumerate(t_values["blur_test"]):
		blurred_img = gaussfft(img, t)

		# Blurred image visualization
		ax = axs2[i]
		ax.imshow(blurred_img, cmap='gray')
		ax.set_title(f't={t}')
		ax.axis('off')

	plt.show()


if exercise == "3.1":
	from Functions import gaussnoise, sapnoise, discgaussfft, medfilt, ideal

	# Create two noisy images 
	office = np.load("Images-npy/office256.npy")
	original = office.copy()
	add = gaussnoise(office, 16)
	sap = sapnoise(office, 0.1, 255)

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(1, 3, 1)
	showgrey(original, False)
	a1.title.set_text("Original image")

	a2 = f.add_subplot(1, 3, 2)
	showgrey(add, False)
	a2.title.set_text("Image with Gaussian noise")

	a3 = f.add_subplot(1, 3, 3)
	showgrey(sap, False)
	a3.title.set_text("Image with salt-and-peppar noise")

	plt.show()


	# 1. Gaussian smoothing
	t_values = [0.1, 0.3, 0.5, 1.0, 2.0, 10.0]

	fig, axs = plt.subplots(2, len(t_values), figsize=(15, 6))
	fig.suptitle("Gaussian Smoothing with Different t Values")
	for i, t in enumerate(t_values):
		# 对带高斯噪声图像进行高斯平滑
		smoothed_add = discgaussfft(add, t)
		axs[0, i].imshow(smoothed_add, cmap='gray')
		axs[0, i].set_title(f"gaussnoise, t={t}")
		axs[0, i].axis('off')

		# 对带椒盐噪声图像进行高斯平滑
		smoothed_sap = discgaussfft(sap, t)
		axs[1, i].imshow(smoothed_sap, cmap='gray')
		axs[1, i].set_title(f"sapnoise, t={t}")
		axs[1, i].axis('off')

	plt.tight_layout()
	plt.show()


	# 2. Median filtering
	window_sizes = [1, 3, 5, 7, 9, 11]

	fig, axs = plt.subplots(2, len(window_sizes), figsize=(15, 6))
	fig.suptitle("Median Filtering with Different Window Sizes")
	for i, w in enumerate(window_sizes):
		# 对带高斯噪声图像进行中值滤波
		smoothed_add = medfilt(add, w)
		axs[0, i].imshow(smoothed_add, cmap='gray')
		axs[0, i].set_title(f"gaussnoise, w={w}")
		axs[0, i].axis('off')

		# 对带椒盐噪声图像进行中值滤波
		smoothed_sap = medfilt(sap, w)
		axs[1, i].imshow(smoothed_sap, cmap='gray')
		axs[1, i].set_title(f"sapnoise, w={w}")
		axs[1, i].axis('off')

	plt.tight_layout()
	plt.show()

	# 3. Ideal low-pass filtering
	cutoff_frequencies = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

	fig, axs = plt.subplots(2, len(cutoff_frequencies), figsize=(15, 6))
	fig.suptitle("Ideal Low-Pass Filtering with Different Cut-Off Frequencies")
	for i, c in enumerate(cutoff_frequencies):
		# 对带高斯噪声图像进行理想低通滤波
		smoothed_add = ideal(add, c)
		axs[0, i].imshow(smoothed_add, cmap='gray')
		axs[0, i].set_title(f"gaussnoise, cut-off={c}")
		axs[0, i].axis('off')

		# 对带椒盐噪声图像进行理想低通滤波
		smoothed_sap = ideal(sap, c)
		axs[1, i].imshow(smoothed_sap, cmap='gray')
		axs[1, i].set_title(f"sapnoise, cut-off={c}")
		axs[1, i].axis('off')

	plt.tight_layout()
	plt.show()


if exercise == "3.2":
	from gaussfft import gaussfft
	from Functions import ideal, rawsubsample

	img = np.load("Images-npy/phonecalc256.npy") 
	smoothimg_gaussian = img.copy()
	smoothimg_ideal = img.copy()
    
	N = 5  # Number of downsampling
	# Filter parameters
	gaussian_t = 4
	ideal_cutoff = 0.2

	f = plt.figure()
	f.subplots_adjust(wspace=0, hspace=0)
	for i in range(N):
		if i > 0: # generate subsampled versions
			img = rawsubsample(img)
			smoothimg_gaussian = gaussfft(smoothimg_gaussian, gaussian_t)
			smoothimg_gaussian = rawsubsample(smoothimg_gaussian)
			smoothimg_ideal = ideal(smoothimg_ideal, ideal_cutoff)
			smoothimg_ideal = rawsubsample(smoothimg_ideal)

		f.add_subplot(3, N, i + 1)
		showgrey(img, False)
		f.add_subplot(3, N, i + N + 1)
		showgrey(smoothimg_gaussian, False)
		f.add_subplot(3, N, i + 2*N + 1)
		showgrey(smoothimg_ideal, False)
	plt.show()
