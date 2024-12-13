import numpy as np
import matplotlib.pyplot as plt

from homography import *
from utils import generate_2d_points, extract_and_match_SIFT

task = "2"


if task == "1":
    num_run = 5
    noise_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    noutliers = [0, 1, 2, 3, 4, 5, 6, 7]
    focal = 1000

    for _ in range(num_run):
        for noise in noise_level:
            pts1, pts2, H = generate_2d_points(num = 100, noutliers = 0, noise = noise, focal = focal)
            draw_matches(pts1, pts2)

            print("Run %s, Error rate: %s" % (_+1, noise))
            print('True H =\n', H)
            H2 = find_homography(pts1, pts2)
            print('Estimated H =\n', H2)
            print('Error =', homography_error(H, H2, focal))
            print("---------------------------------------")

    # --------------------------------------------- #

    for _ in range(num_run):
        for num in noutliers:
            pts1, pts2, H = generate_2d_points(num = 100, noutliers = num, noise = 0.5, focal = focal)
            #draw_matches(pts1, pts2)

            print("Run %s, Number of outliers: %s" % (_+1, num))
            # print('True H =\n', H)
            H2 = find_homography(pts1, pts2)
            # print('Estimated H =\n', H2)
            print('Error =', homography_error(H, H2, focal))
            print("---------------------------------------")


if task == "2":
    num_points = 100
    noutliers = 50
    noise = 0.5
    focal = 1000
    threshold = 1.0

    # 迭代次数范围
    iteration_counts = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    runs_per_iteration = 5  # 每个迭代次数运行多次实验

    # 保存实验结果
    results = []

    for niter in iteration_counts:
        errors = []
        inliers = []
        for _ in range(runs_per_iteration):
            # 生成匹配点
            pts1, pts2, true_H = generate_2d_points(num=num_points, noutliers=noutliers, noise=noise, focal=focal)

            # RANSAC 计算
            Hbest, ninliers, _ = find_homography_RANSAC(pts1, pts2, niter=niter, thresh=threshold)

            # 记录误差和内点数量
            error = homography_error(true_H, Hbest, focal)
            errors.append(error)
            inliers.append(ninliers)

        # 保存当前迭代次数的平均误差和内点数量
        results.append((niter, np.mean(errors), np.mean(inliers)))

    # 提取数据
    niter_values = [r[0] for r in results]
    error_values = [r[1] for r in results]
    inlier_ratios = [r[2] / num_points for r in results]

    # 绘制误差随迭代次数变化的图
    plt.figure(figsize=(10, 6))
    plt.plot(niter_values, error_values, marker='o')
    plt.title("Impact of Iterations on Homography Error")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Homography Error")
    plt.xscale("log")
    plt.grid()
    plt.show()

    # 绘制内点比例随迭代次数变化的图
    plt.figure(figsize=(10, 6))
    plt.plot(niter_values, inlier_ratios, marker='o', color='green')
    plt.title("Impact of Iterations on Inlier Ratio")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Inlier Ratio")
    plt.xscale("log")
    plt.grid()
    plt.show()

    # 加载图像
    img1 = cv2.imread('images/books1.jpg', 0)  # 使用 books1 图像
    img2 = cv2.imread('images/books2.jpg', 0)  # 使用 books2 图像
    img3 = cv2.imread('images/img1.jpg', 0)  # 使用 img1.jpg
    img4 = cv2.imread('images/img2.jpg', 0)  # 使用 img2.jpg

    # 提取 SIFT 特征点
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num=1000)
    pts3, pts4 = extract_and_match_SIFT(img3, img4, num=1000)

    # 使用 RANSAC 计算单应性矩阵
    niter = 10000  # 迭代次数设为 10000
    threshold = 1.0  # 内点判断阈值
    Hbest, ninliers, errors = find_homography_RANSAC(pts1, pts2, niter=niter, thresh=threshold)
    Hbest2, ninliers2, errors2 = find_homography_RANSAC(pts1, pts2, niter=niter, thresh=threshold)

    # 去除异常点后重新估计 H
    inlier_mask = errors < threshold
    inlier_mask2 = errors2 < threshold
    Hfinal = find_homography_RANSAC(pts1[:, inlier_mask], pts2[:, inlier_mask])[0]
    Hfinal2 = find_homography_RANSAC(pts3[:, inlier_mask], pts4[:, inlier_mask])[0]

    # 输出结果
    print(f'RANSAC 内点数量 = {ninliers}/{pts1.shape[1]}')
    print(f'RANSAC 估计的 H =\n{Hbest}')
    print(f'最终估计的 H =\n{Hfinal}')


    def plot_matches_with_lines(img1, img2, pts1, pts2, mask, max_lines=200):
        img_combined = np.hstack((img1, img2))
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(img_combined, cmap='gray')

        # 随机选择 max_lines 条连线
        num_points = min(len(mask), max_lines)
        selected_indices = np.random.choice(len(mask), num_points, replace=False)

        for i in selected_indices:
            x1, y1 = pts1[:, i]
            x2, y2 = pts2[:, i] + np.array([img1.shape[1], 0])
            color = np.random.rand(3, )  # 随机颜色
            ax.plot([x1, x2], [y1, y2], color=color, lw=1)
            ax.scatter(x1, y1, c=[color], s=20)
            ax.scatter(x2, y2, c=[color], s=20)

        ax.set_title("Feature Matches with Random Colors")
        plt.axis("off")
        plt.show()


    plot_matches_with_lines(img1, img2, pts1, pts2, errors < threshold)
    plot_matches_with_lines(img3, img4, pts3, pts4, errors2 < threshold)

    # 可视化融合后的图像
    draw_homography(img1, img2, Hfinal)
    draw_homography(img3, img4, Hfinal2)

