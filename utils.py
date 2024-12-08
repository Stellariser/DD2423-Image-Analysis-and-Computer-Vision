import numpy as np
import cv2

def generate_2d_points(num:int = 500, noutliers:int = 100, noise:float = 1.0, vergence:float = 10.0, 
                       focal:float = 1000) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Generate synthetic points on a planar surface, then projecting them on two different cameras with known relation.

        :param int num: the number of points to generate.
        :param int noutliers: the number of points to convert into outliers.
        :param float noise: the maximum noise (in pixels) added to the projections to add realism.
        :param float vergence: the vergence between the two cameras, in degrees.
        :param float focal: the focal length of the two cameras.
        :returns np.ndarray pts1, np.ndarray pts2, np.ndarray H:
            pts1: a 2xnum array containing the coordinates of the points projections in the first camera.
            pts2: a 2xnum array containing the coordinates of the points projections in the second camera. Points are indexed as in pts1.
            H: a 3x3 matrix containing the theoretical homography between the first and the second camera.
    '''
    pts3d = random_3d_points(num = num, spread = 1.0)
    pts3d[2,:] = 0
    P1, P2 = generate_projections(distance = 6.0, vergence = vergence, focal = focal)
    pts1 = project_points(P1, pts3d) + np.random.randn(2, num)*noise
    pts2 = project_points(P2, pts3d) + np.random.randn(2, num)*noise
    pts2 = add_outliers(pts2, num = noutliers, spread = focal/10)
    H = np.eye(3)
    H[2,0] = +2.0*np.tan(np.pi*vergence/2.0/180.0)/focal
    return pts1, pts2, H

def generate_3d_points(num:int = 500, noutliers:int = 100, noise:float = 1.0, vergence:float = 10.0, focal:float = 1000, 
                       spherical:bool = True) ->tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Generate synthetic points on 3d space, then projecting them on two different cameras with known relation.

        :param int num: the number of points to generate.
        :param int noutliers: the number of points to convert into outliers.
        :param float noise: the maximum noise (in pixels) added to the projections to add realism.
        :param float vergence: the vergence between the two cameras, in degrees.
        :param float focal: the focal length of the two cameras.
        :param bool spherical: if True, generate points on a sphere.
        :returns np.ndarray pts1, np.ndarray pts2, np.ndarray F:
            pts1: a 2xnum array containing the coordinates of the points projections in the first camera.
            pts2: a 2xnum array containing the coordinates of the points projections in the second camera. Points are indexed as in pts1.
            F: a 3x3 matrix containing the theoretical F matrix between the first and the second camera.
    '''
    pts3d = random_3d_points(num = num, spread = 1.0, spherical = spherical)
    P1, P2 = generate_projections(distance = 6.0, vergence = vergence, focal = focal)
    F = fmatrix_from_projections(P1, P2)
    pts1 = project_points(P1, pts3d) + np.random.randn(2, num)*noise
    pts2 = project_points(P2, pts3d) + np.random.randn(2, num)*noise
    pts2 = add_outliers(pts2, num = noutliers, spread = focal/10)
    return pts1, pts2, F

def extract_and_match_SIFT(img1, img2, max_distance = 10000.0, num = 10000):
    '''Extract and matches SIFT features in img1 and img2, keeping the best num matches.
        :returns np.ndarray pts1, np.ndarray pts2:
            pts1: a 2x(min(num,macthed_kp)) array containing the keypoints from img1.
            pts2: a 2x(min(num,matched_kp)) array containing the keypoints from img2. Matching points in pts1 and pts2 have corresponding indexes.
    '''
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # detect and compute the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    mpairs = bf.knnMatch(des1, des2, k = 2)
    matches = []
    for m1,m2 in mpairs:
        if m1.distance < 0.9*m2.distance and m1.distance<max_distance:
            matches.append(m1)
    # sort the matches based on distance
    matches = sorted(matches, key=lambda val: val.distance)
    # store in numpy arrays
    n = min(len(matches), num)
    pts1 = np.zeros((2, n))
    pts2 = np.zeros((2, n))
    for i in range(n):
        pts1[:,i] = kp1[matches[i].queryIdx].pt
        pts2[:,i] = kp2[matches[i].trainIdx].pt 
    return pts1, pts2

def random_3d_points(num, spread = 1.0, spherical = False):
    pts = np.random.uniform(-spread, spread, (3, num))
    if spherical:
        rad = np.sqrt(np.sum(pts**2,0))
        pts = pts/(rad + 1e-12)
    return pts

def generate_projections(distance, vergence, focal): # focal in pixels
    angle2 = vergence/180.0*np.pi/2.0
    ca = np.cos(angle2)
    sa = np.sin(angle2)
    P1 = np.array([[ ca*focal, 0.0,  sa*focal, 0.0 ], [ 0.0, focal, 0.0, 0.0 ], [ -sa, 0.0, ca, distance ]])
    P2 = np.array([[ ca*focal, 0.0, -sa*focal, 0.0 ], [ 0.0, focal, 0.0, 0.0 ], [  sa, 0.0, ca, distance ]])
    return P1, P2

def project_points(P, pts3d): 
    pts = P @ np.vstack([pts3d, np.ones((1,np.size(pts3d, 1)))])
    pts = pts[0:2,:]/pts[2,:]
    return pts

def add_outliers(pts, num, spread = 100): # spread in pixels
    npts = np.size(pts, 1)
    ids = np.random.choice(npts, num, replace=False)
    pts[:,ids] +=  np.random.uniform(-spread, spread, (2,num))
    return pts

def fmatrix_from_projections(p1, p2):
    fmatrix = np.zeros((3, 3))
    x = np.empty((3, 2, 4))
    x[0, :, :] = np.vstack([p1[1, :], p1[2, :]])
    x[1, :, :] = np.vstack([p1[2, :], p1[0, :]])
    x[2, :, :] = np.vstack([p1[0, :], p1[1, :]])
    y = np.empty((3, 2, 4))
    y[0, :, :] = np.vstack([p2[1, :], p2[2, :]])
    y[1, :, :] = np.vstack([p2[2, :], p2[0, :]])
    y[2, :, :] = np.vstack([p2[0, :], p2[1, :]])
    for i in range(3):
        for j in range(3):
            xy = np.vstack([x[j, :], y[i, :]])
            fmatrix[i, j] = np.linalg.det(xy)
    return fmatrix