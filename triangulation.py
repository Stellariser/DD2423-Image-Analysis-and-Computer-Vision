import numpy as np
import cv2
from utils import extract_and_match_SIFT
from plots import get_keypoint_colors,draw_matches,draw_2d_points,draw_3d_points,draw_triangles
from fmatrix import find_fmatrix,find_fmatrix_RANSAC

def triangulate(M1:np.ndarray, M2:np.ndarray, pts1:np.ndarray, pts2:np.ndarray)->np.ndarray:
    '''Triangulate two set of points pts1 and pts2 (with corresponding matrices M1 and M2) into 3d points.

        :param np.ndarray M1: the 3x4 projection matrix for the first camera.
        :param np.ndarray M2: the 3x4 projection matrix for the second camera.
        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :returns np.ndarray: a 3xN_points array containing 3d coordinates of triangulated points.

    '''
    n = pts1.shape[1]  # Number of points
    pts3d = np.zeros((3, n))  # To store the 3D points

    for i in range(n):
        x1, y1 = pts1[:, i]
        x2, y2 = pts2[:, i]

        G = np.array([
            x1 * M1[2, :] - M1[0, :],
            y1 * M1[2, :] - M1[1, :],
            x2 * M2[2, :] - M2[0, :],
            y2 * M2[2, :] - M2[1, :]
        ])

        # Solve using SVD
        U, S, Vt = np.linalg.svd(G)
        X = Vt[-1, :]  # Solution corresponding to the smallest singular value
        X /= X[-1]  # Convert from homogeneous to Euclidean coordinates

        # Store the 3D point
        pts3d[:, i] = X[:3]

    return pts3d


def projection_from_fmatrix(F, pts1, pts2, focal = 1000):
    # remove the camera matrices, essentially change focal length to one
    Fn = np.diag([focal, focal, 1.0]) @ F @ np.diag([focal, focal, 1.0])
    pt1 = pts1/focal
    pt2 = pts2/focal
    # find four possible projection matrices
    U, S, Vh = np.linalg.svd(Fn, full_matrices=True)
    t = Vh[2,:].reshape(-1,1)
    W = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    R1 = U @ W.T @ Vh
    R2 = U @ W @ Vh
    R1 = R1 * np.linalg.det(R1)
    R2 = R2 * np.linalg.det(R2)
    M2 = [ R1 @ (np.hstack((np.eye(3),  t))), 
        R1 @ (np.hstack((np.eye(3), -t))),
        R2 @ (np.hstack((np.eye(3),  t))),  
        R2 @ (np.hstack((np.eye(3), -t))) ]
    # test four alternative solutions
    maxn = -1
    M1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    for i in range(4):
        p1 = triangulate(M1, M2[i], pt1, pt2)
        p2 = M2[i] @ np.vstack((p1, np.ones((1, np.size(p1, 1))))) 
        n = np.sum(p1[2,:]>0) + np.sum(p2[2,:]>0)
        if n > maxn:
            M = M2[i]
            pts3d = p1
            maxn = n
    return M, pts3d


def run_triangulation(image_path_1, image_path_2, focal, niter=10000, thresh=1.):
    img1 = cv2.imread(image_path_1, 0)
    img2 = cv2.imread(image_path_2, 0)
    pts1,pts2 = extract_and_match_SIFT(img1, img2, num=1000)

    # center the points for less biased prediction of fmatrix
    h, w = np.shape(img1)
    cpts1 = pts1 - np.array([[w/2],[h/2]])
    h, w = np.shape(img2)
    cpts2 = pts2 - np.array([[w/2],[h/2]])

    # find F matrix and M
    F1, ninliers, errors = find_fmatrix_RANSAC(cpts1, cpts2, niter = niter, thresh = thresh)
    # draw_matches(pts1[:,errors<thresh], pts2[:,errors<thresh], img1, img2)
    F2 = find_fmatrix(cpts1[:,errors<thresh], cpts2[:,errors<thresh], normalize = True)
    M, pts3d = projection_from_fmatrix(F2, cpts1[:,errors<thresh], cpts2[:,errors<thresh], focal)
    rotation_angle = np.arccos((np.abs(M[0,0] + M[1,1] + M[2,2]) - 1.0)/2.0)*180.0/np.pi
    # print('\nnumber of features: ', np.size(pts1, 1))
    # print('number of inliers: ', ninliers)
    # print('initial F = \n', F1/np.linalg.norm(F1))
    # print('final F = \n', F2/np.linalg.norm(F2))
    # print('projection M = \n', np.diag((focal, focal, 1)) @ M)
    # print('rotation angle = ', rotation_angle)

    #plots
    # real_colors = get_keypoint_colors(image_path_1, pts1)
    # depth_colors = (pts3d[2,:] - np.min(pts3d[2,:]))/(np.max(pts3d[2,:]) - np.min(pts3d[2,:]))
    # draw_2d_points(cv2.cvtColor(cv2.imread(image_path_1,1), cv2.COLOR_BGR2RGB), pts1[:,errors<thresh], depth_colors)
    # draw_3d_points(pts3d,(real_colors[:,errors<thresh]).T)
    # draw_triangles(pts1[:,errors<thresh], pts3d)
    return rotation_angle

if __name__=="__main__":
    np.set_printoptions(precision = 3)

    #TODO: You can use this function to perform your tests or try our examples

    ## Task 5 example
    focal = 1000
    run_triangulation("images/books1.jpg", "images/books2.jpg", focal)