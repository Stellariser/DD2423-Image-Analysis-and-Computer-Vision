import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt

def draw_matches(pts1, pts2, img1 = None, img2 = None, width = 0):
    '''Shows matches pts1 and pts2 above the images img1,img2
    '''
    fig, ax = plt.subplots()
    if img1 is not None:
        width = np.size(img1, 1)
        gap = np.ones((max(np.size(img1, 0),np.size(img2, 0)), int(width/4)))*255
        dh=np.abs(np.size(img1, 0)-np.size(img2, 0))
        if np.size(img1, 0)<np.size(img2, 0):
            ax.imshow(np.hstack((np.vstack((img1,np.ones((dh,np.size(img1, 1)))*255)), gap, img2)), cmap='gray')  
        elif np.size(img1, 0)<np.size(img2, 0):
            ax.imshow(np.hstack((img1, gap,np.vstack((img2,np.ones((dh,np.size(img2, 1)))*255)))), cmap='gray')  
        else:
            ax.imshow(np.hstack((img1, gap, img2)), cmap='gray')        
    for i in range(np.size(pts1,1)):
        ax.plot([pts1[0,i], pts2[0,i] + width*5/4], [pts1[1,i], pts2[1,i]], c=None, lw=0.75)
    ax.scatter(pts1[0,:], pts1[1,:], c='b', s=4.0**2) 
    ax.scatter(pts2[0,:] + width*5/4, pts2[1,:], c='r', s=4.0**2) 
    fig.tight_layout()
    plt.show()  

def draw_homography(img1, img2, H, save = False):
    ''' Overlaps img1 and img2 according to the homography H 
    '''
    invH = np.linalg.inv(H)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, invH)
    pts = np.concatenate((pts1, pts2_), axis=0)
    #Finding the minimum and maximum coordinates
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    #Warping the first image on the second image using Homography Matrix
    result = np.zeros((ymax-ymin, xmax-xmin, 3), np.uint8)
    result[:,:,2] = cv2.warpPerspective(img2, Ht.dot(invH), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0], 0] = img1
    result[t[1]:h1+t[1], t[0]:w1+t[0], 1] = img1
    if save:
        cv2.imwrite('homography.jpg', result) #Uncomment to save the image
    plt.imshow(result)
    plt.axis('off')
    plt.show()
    return

def draw_2d_points(img,pts,colors=None):
    fig, ax = plt.subplots()
    if img.squeeze().ndim==2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    if colors is not None:
        ax.scatter(pts[0,:], pts[1,:], c=colors, s=2.0**2)
    else:
        ax.scatter(pts[0,:], pts[1,:], 'r', s=2.0**2)
    fig.tight_layout()
    plt.show()

def draw_3d_points(pts3d, colors=None):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    if colors is None:
        ax.scatter3D(pts3d[0,:], pts3d[1,:], pts3d[2,:], c = 'r')
    else:
        ax.scatter3D(pts3d[0,:], pts3d[1,:], pts3d[2,:], c = colors)
    plt.title("3D Reconstruction")
    plt.show()

def get_keypoint_colors(img_path, pts, ws = 3):
    ''' Gets RGB colors of keypoints from two images and the list of keypoints.
    '''
    img1_color = cv2.cvtColor(cv2.imread(img_path,1), cv2.COLOR_BGR2RGB)
    
    colors = np.zeros((3, pts.shape[1]))
    idx1 = np.trunc(pts).astype(np.int64)
    min_x = np.clip(idx1[0,:]-ws,a_min=0,a_max=img1_color.shape[1]).astype(np.int64)
    max_x = np.clip(idx1[0,:]+ws,a_min=0,a_max=img1_color.shape[1]).astype(np.int64)
    min_y = np.clip(idx1[1,:]-ws,a_min=0,a_max=img1_color.shape[0]).astype(np.int64)
    max_y = np.clip(idx1[1,:]+ws,a_min=0,a_max=img1_color.shape[0]).astype(np.int64)
    for i in range(pts.shape[1]):
        colors[:,i] = (img1_color[min_y[i]:max_y[i],min_x[i]:max_x[i]]).mean(axis=(0,1))
    if np.sum(colors>1)>0:
        colors /= 255
    return colors

def draw_triangles(pts2d, pts3d):
    tri = scipy.spatial.Delaunay(pts2d.T)
    ax = plt.axes(projection="3d")
    center = np.median(pts3d, axis=1)
    spread = 0.55*np.max(np.percentile(pts3d, 98.0, axis=1) - np.percentile(pts3d, 2.0, axis=1))
    ax.set_xlim3d(center[0]-spread, center[0]+spread)
    ax.set_ylim3d(center[1]-spread, center[1]+spread)
    ax.set_zlim3d(center[2]-spread, center[2]+spread)
    ax.plot_trisurf(pts3d[0,:], pts3d[1,:], pts3d[2,:], triangles = tri.simplices)        
    plt.title("3D Reconstruction")
    plt.show()