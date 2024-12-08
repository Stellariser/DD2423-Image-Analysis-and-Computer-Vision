import numpy as np
import cv2
from utils import extract_and_match_SIFT,generate_3d_points
from plots import draw_matches

def find_fmatrix(pts1:np.ndarray, pts2:np.ndarray, normalize:bool = False) -> np.ndarray:
    '''Estimate the F matrix from the matching points pts1 and pts2

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes. 
        :param bool normalize: if True, points are normalized to improve stability.
        :returns np.ndarray: a 3x3 array representing the F matrix.

    '''
    # For better stability, normalize points to be centered at (0,0) with unit variance 
    if normalize:
        mean1 = np.mean(pts1, axis=1)
        std1 = np.std(pts1, axis=1)
        T1 = np.array([[1/std1[0], 0, -mean1[0]/std1[0]], [0, 1/std1[1], -mean1[1]/std1[1]], [0, 0, 1]])
        pts1 = T1 @ np.vstack((pts1, np.ones((1, np.size(pts1, 1)))))
        mean2 = np.mean(pts2, axis=1)
        std2 = np.std(pts2, axis=1)
        T2 = np.array([[1/std2[0], 0, -mean2[0]/std2[0]], [0, 1/std2[1], -mean2[1]/std2[1]], [0, 0, 1]])
        pts2 = T2 @ np.vstack((pts2, np.ones((1, np.size(pts2, 1)))))
 
    #TODO: ADD YOUR CODE HERE (and remove next line)
    raise NotImplementedError("F matrix estimation yet to be done. Complete the function find_fmatrix and remove this line.")

    # Undo the normalization to get F in the original coordinates
    if normalize:
        F = T2.T @ F @ T1
    return F


def fmatrix_error(F1:np.ndarray, F2:np.ndarray, focal:float) -> float:
    '''Computes the error between two F matrices.
        :param np.ndarray F1: a 3x3 matrix representing one of the F matrices.
        :param np.ndarray F2: a 3x3 matrix representing the second F matrix.
        :returns float: the error between the two F matrices.
    '''
    F1n = np.diag([focal, focal, 1.0]) @ F1 @ np.diag([focal, focal, 1.0])
    F2n = np.diag([focal, focal, 1.0]) @ F2 @ np.diag([focal, focal, 1.0])
    F1n = F1n/np.linalg.norm(F1n)
    F2n = F2n/np.linalg.norm(F2n)
    if np.sum(F1n*F2n)<0:
        F2n = -F2n
    return np.linalg.norm(F1n - F2n)


def count_fmatrix_inliers(F, pts1, pts2, thresh = 0.5):
    '''Given the matrix F, projects pts1 on the second image, counting the number of actual points in pts2 for which the projection error is smaller than the given threshold.

        :param np.ndarray F: a 3x3 matrix containing the F matrix.
        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param float thresh: the threshold to consider points as inliers.
        :returns int ninliers, np.ndarray errors:
            ninliers: the number of inliers.
            errors: a N_points array containing the errors; they are indexed as pts1 and pts2.
    
    '''
    Fp = F@np.vstack((pts1, np.ones((1,np.size(pts1, 1)))))
    pF = F.T@np.vstack((pts2, np.ones((1,np.size(pts2, 1)))))
    pFp = (Fp[0,:]*pts2[0,:] + Fp[1,:]*pts2[1,:] + Fp[2,:])**2
    l1 = Fp[0,:]**2 + Fp[1,:]**2 
    l2 = pF[0,:]**2 + pF[1,:]**2 
    errors = np.sqrt(pFp/l1 + pFp/l2)
    ninliers = np.sum(np.where(errors<thresh, 1, 0))
    return ninliers, errors


def find_fmatrix_RANSAC(pts1:np.ndarray, pts2:np.ndarray, niter:int = 100, thresh:float = 1.0):
    '''Computes the best F matrix for matching points pts1 and pts2, adopting RANSAC.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param int niter: the number of RANSAC iteration to run.
        :param float thresh: the maximum error to consider a point as an inlier while evaluating a RANSAC iteration.
        :returns np.ndarray Fbest, int ninliers, np.ndarray errors:
            Fbest: a 3x3 matrix representing the best F matrix found.
            ninliers: the number of inliers for the best F matrix found.
            errors: a N_points array containing the errors for the best F matrix found; they are indexed as pts1 and pts2.
    
    '''
    #TODO: ADD YOUR CODE HERE (and remove next line)
    raise NotImplementedError("F matrix RANSAC based estimation yet to be done. Complete the function find_fmatrix_RANSAC and remove this line.")

    return Fbest, ninliers, errors


def synthetic_example(RANSAC=False):
    focal = 1000
    pts1, pts2, F = generate_3d_points(num = 200, noutliers = 20, noise=0.5, focal=focal, spherical = True)
    draw_matches(pts1, pts2) 
    print('True F =\n', F/np.linalg.norm(F))
    if RANSAC:
        F1,ninliers,errors = find_fmatrix_RANSAC(pts1, pts2, niter=10000)
        F2 = find_fmatrix(pts1[:,errors<1], pts2[:,errors<1], normalize=True)
        print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
        print('RANSAC F =\n', F1/np.linalg.norm(F1))
        print('Final estimated F =\n', F2/np.linalg.norm(F2))
    else:
        F2 = find_fmatrix(pts1, pts2, normalize=True) 
        print('Estimated F =\n', F2/np.linalg.norm(F2))
    print('Error =', fmatrix_error(F, F2, focal))


def real_example():
    img1 = cv2.imread('images/desk1.jpg', 0)
    img2 = cv2.imread('images/desk2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    draw_matches(pts1, pts2, img1, img2)
    F1,inliers,errors = find_fmatrix_RANSAC(pts1, pts2, 10000)
    draw_matches(pts1[:,errors<1], pts2[:,errors<1], img1, img2)


if __name__=="__main__":
    np.set_printoptions(precision = 3)

    #TODO: You can use this function to perform your tests or try our examples

    ## Task 3 example
    synthetic_example(RANSAC = False)

    ## Task 4 example (from synthetic data)
    #synthetic_example(RANSAC = True)

    ## Task 4 example (from real images)
    #real_example()