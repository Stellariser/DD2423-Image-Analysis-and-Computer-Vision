import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from scipy.ndimage import rotate
from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


exercise = "2.3"


if exercise == "1.3":
	coordinates = [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]
	for u, v in coordinates:
		fftwave(u, v)


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