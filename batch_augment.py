import numpy as np
import cv2

def get_auged_batch(batch,eps=0.08,shift=3.0):
    if np.random.rand(1) > 0.5:
        I = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
    else:
        I = np.array([[-1.0, 0.0, batch.shape[1]], [0.0, 1.0, 0.0]])
    noise = eps*np.random.randn(2,3)
    s = shift*np.random.randn(2,3)*np.array([[0.0,0.0,1.0],[0.0,0.0,1.0]])
    M = I + noise + s
    warped = []

    for i in range(batch.shape[0]):
        warped.append(cv2.warpAffine(batch[i,:,:,:],M,dsize=batch.shape[1:3]))
    return np.stack(warped)