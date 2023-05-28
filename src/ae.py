import cv2 as cv
import numpy as np,sys
A = cv.imread('/home/luoleyouluole/Image-Restoration-Experiments/src/ae.png')
assert A is not None, "file could not be read, check with os.path.exists()"
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    L = gpA[i-1] - GE + 0.01
    # L = cv.subtract(gpA[i-1],GE)
    # L = cv.add(L,np.ones_like(L) * 0.5)
    lpA.append(L)


# now reconstruct
ls_ = lpA[0]
for i in range(1,6):
    ls_ = cv.pyrUp(ls_)
    ls_ = ls_ + lpA[i]
    # ls_ = cv.add(ls_, lpA[i])
# image with direct connecting each half
cv.imwrite('enhanced.jpg',ls_)
