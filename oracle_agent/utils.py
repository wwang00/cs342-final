  
import numpy as np
import cv2

def mask_to_average(mask):
    xs = np.where(np.any(mask, axis=0))[0]
    if xs.shape[0] == 0:
        return None, None
    ax = np.mean(xs)
    ay = np.mean(np.where(np.any(mask, axis=1))[0])
    return ax, ay

def mask_to_world(mask, proj, height=.36983):
    # get normalized coords from mask
    ax, ay = mask_to_average(mask)
    if ax is None:
        return None
    # assert len(mask.shape) == 2
    # assert proj.shape == (4,4)
    tx = (2*ax/mask.shape[1]) - 1
    ty = 1-(2*ay/mask.shape[0])

    # get projection and inverse projection
    proji = np.linalg.inv(proj)

    
    # get z value (0,1) using est height of puck
    homocam = np.array([tx, ty, 1.0]).reshape(3,1)
    pn2 = np.hstack([ proji[:,:2], proji[:,3:] ])
    zpred = ((pn2[1:2,:] @ homocam) - (height * (pn2[3:4,:] @ homocam))) / (height*proji[3,2] - proji[1,2])
    # zpred = (pn2[1:2,:] @ homocam) / (-proji[1,2]) # this impl assumes height of puck is 0

    # case where point is behind us, i.e., predicted height is wrong
    if zpred < 0:
        return None

    # project onto world coords
    bighomocam = np.array([tx, ty, float(zpred), 1.0]).reshape(4,1)
    world_pred = proji @ bighomocam
    world_pred /= world_pred[3]
    
    return world_pred.flatten()[:3]

def draw_cords(image, world_cord, proj):

    c4d = np.vstack([world_cord.reshape(3,1), np.array([[1.0]])] )

    cam_proj = proj @ c4d
    cam_proj /= cam_proj[3]
    
    # if dot out of focal range, or behind us
    if abs(cam_proj[0]) > 1 or abs(cam_proj[1]) > 1 or c4d[2] < 0:
        return image
    # convert to image coords wrt top left corner
    ix = int(image.shape[1]/2 + cam_proj[0] * image.shape[1]/2)
    iy = int(image.shape[0]/2 - cam_proj[1] * image.shape[0]/2)
    return cv2.circle(image,(ix, iy), 3, (0,255,0), -1)

def draw_arr(image, arr, ht=20):
    s = ''
    for i in range(arr.shape[0]):
        s += f'{float(arr[i]):.2f},'
    return cv2.putText(img=image, text=s, org=(ht,30),fontFace=5, fontScale=1, color=(0,0,255), thickness=1)
