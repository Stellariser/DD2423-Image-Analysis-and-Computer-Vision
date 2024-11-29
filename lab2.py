import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft
        
exercise = "6"

# 1 Q1
# 2 Q2,3
# 4 Q4，5，6
# 5 Q7
# 6 Q8,9,10

def deltax():
    # Sobel operator
    # A derivative is a highpass filter and will enhance noise.
    # The first order derivative varies due to the position of the light source.
    # Sobel operator add some smoothing, but do it in the opposite direction.
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
    # 梯度大小表示圖像像素值變化的強度，也就是像素值變化最劇烈的程度。
    Lx = convolve2d(inpic, deltax(), shape)
    Ly = convolve2d(inpic, deltay(), shape)
    return np.sqrt(Lx**2 + Ly**2)

def Lvvtilde(inpic, shape = 'same'):
    # Introduce in each point a local coordinate system (u, v) such that the v direction is parallel to the gradient direction

    # Lvvtilde = 0 indicates a zero-crossing point in the second-order directional derivative along the gradient direction. 
    # This condition identifies locations where the intensity changes most rapidly, which are potential edge points.
    # However, zero-crossings alone may include noise or irrelevant features that do not correspond to meaningful edges.

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
    # Introduce in each point a local coordinate system (u, v) such that the v direction is parallel to the gradient direction

    # Lvvvtilde < 0 ensures that the zero-crossing point corresponds to a concave curvature in the intensity variation.
    # This means that the curvature decreases at this point, indicating a transition in intensity that is more likely to be a true edge.
    # By combining this condition with Lvvtilde = 0, we can filter out noise-induced zero-crossings and retain only significant edges.

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
    # Step 1: Smooth the input image using Gaussian smoothing
    smoothed_image = discgaussfft(inpic, scale)

    # Step 2: Compute the second-order directional derivative (Lvv~)
    Lvv = Lvvtilde(smoothed_image, shape)

    # Step 3: Compute the third-order directional derivative (Lvvv~)
    Lvvv = Lvvvtilde(smoothed_image, shape)

    # Step 4: Create a mask from the sign condition (Lvvv < 0)
    Lvvv_mask = (Lvvv < 0)

    # Step 5: Detect zero-crossing curves in Lvv~, filtered by the Lvvv mask
    edge_curves = zerocrosscurves(Lvv, Lvvv_mask)

    # Step 6: Compute the gradient magnitude of the smoothed image
    gradient_magnitude = Lv(smoothed_image, shape)

    # Step 7: Create a mask by thresholding the gradient magnitude
    gradient_mask = (gradient_magnitude > threshold)

    # Step 8: Refine the zero-crossing curves using the gradient magnitude mask
    edge_curves = thresholdcurves(edge_curves, gradient_mask)

    return edge_curves
        
def houghline(curves, magnitude, nrho, ntheta, threshold, nlines = 20, verbose = False, increment_func=lambda x: 1):
    """
    Performs the Hough transform for detecting straight lines in an image.

    Arguments:
    - curves: Edge points as a tuple (Y, X) from edge detection.
    - magnitude: Gradient magnitude image.
    - nrho: Number of accumulators in the rho direction.
    - ntheta: Number of accumulators in the theta direction.
    - threshold: Minimum magnitude value for valid points.
    - nlines: Number of strongest lines to extract.
    - verbose: Denotes the degree of extra information and figures that will be shown.

    Returns:
    - linepar: A list of (ρ, θ) parameters for each line segment [(rho1, theta1), (rho2, theta2), ...].
    - acc: Accumulator matrix of the Hough transform.
    """
    # Step 1: Allocate accumulator space
    acc = np.zeros((nrho, ntheta))  # Accumulator array for (rho, theta)
    thetas = np.linspace(-np.pi/2, np.pi/2, ntheta)  # Theta values in radians
    diag_len = np.sqrt(magnitude.shape[0]**2 + magnitude.shape[1]**2)  # 計算圖像的對角線長度（圖像的最大可能 rho 值）
    rhos = np.linspace(-diag_len, diag_len, nrho)  # Rho values

    # Step 2: Loop over all edge points
    Y, X = curves  # Extract edge points
    for y, x in zip(Y, X):
        # Check if the gradient magnitude exceeds the threshold
        grad_value = magnitude[int(y), int(x)]
        if grad_value > threshold:
            # Loop through theta values to compute rho
            for theta_idx, theta in enumerate(thetas):
                rho = x * np.cos(theta) + y * np.sin(theta)  # Compute rho
                # Map rho to its index in the accumulator
                rho_idx = int(np.round((rho + diag_len) * (nrho - 1) / (2 * diag_len)))
                # Update the accumulator
                acc[rho_idx, theta_idx] += increment_func(grad_value)

    # Step 3: Find local maxima in the accumulator
    flat_indices = np.argsort(-acc.flatten())  # Sort accumulator values descending
    strongest_indices = flat_indices[:nlines]  # Top nlines indices

    # Step 4: Map the strongest responses to (rho, theta)
    linepar = []
    for idx in strongest_indices:
        rho_idx, theta_idx = np.unravel_index(idx, acc.shape)  # Map back to 2D indices
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        linepar.append((rho, theta))

    if verbose:
        print(f"Accumulator shape: {acc.shape}")
        print(f"Detected lines: {linepar}")

    return linepar, acc

def linear_increment(grad_value):
    return grad_value

def squared_increment(grad_value):
    return grad_value ** 2

def log_increment(grad_value):
    return np.log(1 + grad_value)


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines = 20, verbose = False):
    # Extract edges and compute gradient magnitude
    curves = extractedge(pic, scale, gradmagnthreshold, 'same')  # Edge detection
    magnitude = Lv(pic, 'same')  # Compute gradient magnitude

    # Perform Hough transform and get line parameters
    linepar, acc = houghline(curves, magnitude, nrho, ntheta, gradmagnthreshold, nlines, verbose)

    # Diagonal length of the image (used for visualization limits)
    diag_len = np.sqrt(pic.shape[0]**2 + pic.shape[1]**2)

    # Visualization based on verbose level
    if verbose == 0:
        print(linepar)
        print(acc)

    elif verbose == 1:
        f = plt.figure()
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=10)

        a1 = f.add_subplot(1, 3, 1)
        showgrey(pic, False)
        a1.set_title("Original Image") 

        a2 = f.add_subplot(1, 3, 2)
        overlaycurves(pic, curves)
        a2.set_title("Overlay Curves: Image + Curves") 

        a3 = f.add_subplot(1, 3, 3)
        showgrey(magnitude, False)
        a3.set_title("Gradient Magnitude") 
        
        plt.show()


    # Return detected line parameters and accumulator
    return linepar, acc
         




if exercise == "1":
    tools = np.load("Images-npy/few256.npy")
    dxtools = convolve2d(tools, deltax(), 'valid')  #no  padding while valid
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


if exercise == "4":
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


if exercise == "5":
    house = np.load("Images-npy/godthem256.npy")
    tools = np.load("Images-npy/few256.npy")

    scales = [0.0001, 1.0, 4.0, 16.0, 64.0]
    
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    
    ax = f.add_subplot(2, 3, 1)
    showgrey(tools, False)
    ax.set_title(f"Original Image")
    for i, scale in enumerate(scales):
        ax = f.add_subplot(2, 3, i + 2)
        edgecurves = extractedge(tools, scale, 0, 'same')
        overlaycurves(tools, edgecurves)
        ax.set_title(f"Scale = {scale}")
    
    plt.show()
    # --------------------------------------------- #
    thresholds = [1, 3, 5, 7, 9, 11]

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f.suptitle("Thresholded Images with Scale = 4", fontsize=16)
    
    for i, threshold in enumerate(thresholds):
        ax = f.add_subplot(2, 3, i + 1)
        edgecurves = extractedge(tools, 4, threshold, 'same')
        overlaycurves(tools, edgecurves)
        ax.set_title(f"Threshold = {threshold}")
    
    plt.show()
    # --------------------------------------------- #
    # Best results:
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    f.suptitle("Best results", fontsize=16)
    
    a1 = f.add_subplot(1, 2, 1)
    edgecurves_house = extractedge(house, 4, 7, 'same')
    overlaycurves(house, edgecurves_house)
    a1.set_title("House with scale=4 and threshold=7")

    a2 = f.add_subplot(1, 2, 2)
    edgecurves_tools = extractedge(tools, 4, 7, 'same')
    overlaycurves(tools, edgecurves_tools)
    a2.set_title("Tools with scale=4 and threshold=7")
    
    plt.show()

import time
if exercise == "6":
    # 加载实验图像
    testimage = np.load("Images-npy/godthem256.npy")

    nrho_values = [50, 100, 180, 300]
    ntheta_values = [50, 100, 180, 300]

    # 测试结果存储
    results = []

    for nrho, ntheta in zip(nrho_values, ntheta_values):
        start_time = time.time()  # 开始计时
        linepar, acc = houghedgeline(
            pic=testimage,
            scale=2,
            gradmagnthreshold=10,
            nrho=nrho,
            ntheta=ntheta,
            nlines=10,
            verbose=False
        )
        elapsed_time = time.time() - start_time  # 计算时间
        results.append((nrho, ntheta, elapsed_time))

        # 打印累积器空间热图（用于分析）
        plt.figure(figsize=(8, 6))
        plt.title(f"Hough Accumulator (nrho={nrho}, ntheta={ntheta})")
        plt.imshow(acc, cmap='hot', aspect='auto')
        plt.colorbar(label="Accumulator Value")
        plt.xlabel("Theta (radians)")
        plt.ylabel("Rho (pixels)")
        plt.tight_layout()
        plt.show()

    # 打印结果
    for nrho, ntheta, elapsed_time in results:
        print(f"nrho={nrho}, ntheta={ntheta}, time={elapsed_time:.2f}s")

    # 参数配置
    scale = 3
    gradmagnthreshold = 8
    nrho = 360
    ntheta = 360
    nlines = 25
    verbose = 2  # 可视化模式

    # 定义增量函数
    increment_funcs = {
        "Constant Increment (1)": lambda x: 1,
        "Linear Increment (|∇L|)": lambda x: x,
        "Squared Increment (|∇L|^2)": lambda x: x**2,
        "Log Increment (log(1+|∇L|))": lambda x: np.log(1 + x)
    }

    # 提取边缘曲线
    edge_curves = extractedge(testimage, scale, gradmagnthreshold, 'same')
    magnitude = Lv(testimage, 'same')

    # 可视化不同增量函数的效果
    for name, func in increment_funcs.items():
        print(f"Using {name}")
        # 调用 Hough 变换
        linepar, acc = houghline(
            curves=edge_curves,
            magnitude=magnitude,
            nrho=nrho,
            ntheta=ntheta,
            threshold=gradmagnthreshold,
            nlines=nlines,
            verbose=False,
            increment_func=func
        )

        # 提取累积器峰值
        pos, values, _ = locmax8(acc)
        strongest_peaks_idx = np.argsort(values)[-nlines:]  # 获取最强的 nlines 个峰值
        strongest_pos = pos[strongest_peaks_idx]

        # 可视化检测结果
        plt.figure(figsize=(12, 6))

        # **1. 原始图像和检测到的线条**
        plt.subplot(1, 2, 1)
        plt.title(f"Detected Lines ({name})")
        plt.imshow(testimage, cmap='gray')

        # 图像范围
        height, width = testimage.shape
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Y轴方向反转

        # 绘制检测到的线条
        colors = plt.cm.jet(np.linspace(0, 1, nlines))  # 为线条分配不同颜色
        for idx, (rho, theta) in enumerate(linepar):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            dx = width * (-b)
            dy = height * a
            x1, y1 = x0 - dx, y0 - dy
            x2, y2 = x0 + dx, y0 + dy
            plt.plot([x1, x2], [y1, y2], color=colors[idx], linewidth=2, label=f"Line {idx + 1}")

        plt.legend(loc='lower left')

        # **2. 累积器矩阵及峰值标记**
        plt.subplot(1, 2, 2)
        plt.title(f"Hough Accumulator with Peaks Highlighted ({name})")
        plt.imshow(acc, cmap='hot', extent=[
            -np.pi / 2, np.pi / 2,  # Theta 范围
            np.sqrt(testimage.shape[0] ** 2 + testimage.shape[1] ** 2),  # Rho 正范围
            -np.sqrt(testimage.shape[0] ** 2 + testimage.shape[1] ** 2)  # Rho 负范围
        ], aspect='auto')
        plt.colorbar(label="Accumulator Value")
        plt.xlabel("Theta (radians)")
        plt.ylabel("Rho (pixels)")

        # 在累积器矩阵上标记峰值
        for idx, (theta_idx, rho_idx) in enumerate(strongest_pos):
            rho = (rho_idx / (nrho - 1)) * 2 * np.sqrt(testimage.shape[0] ** 2 + testimage.shape[1] ** 2) - \
                  np.sqrt(testimage.shape[0] ** 2 + testimage.shape[1] ** 2)
            theta = (theta_idx / (ntheta - 1)) * np.pi - np.pi / 2
            plt.plot(theta, rho, 'o', color=colors[idx], markersize=8, label=f"Line {idx + 1}")

        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()





