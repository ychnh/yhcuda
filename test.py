import numpy as np
import numpy.testing as npt
import cv2 as cv
import yhcuda

def test():
    H = np.array( [ [.98358, -.002393, -.429099], [.007420, .950114, -.018871], [.045492, .094473, 1] ]).astype(np.float32)
    print(H)
    H = H.reshape(-1)
    img1 = cv.imread('img0.png')
    img2 = cv.imread('img1.png')
    height, width, _ = img1.shape
    img1 = img1.reshape( -1 )
    img2 = img2.reshape( -1 )

    output = yhcuda.c_mov_obj_detect(img1, img2, H, width, height)
    output = output.reshape((height, width, 3))
    print(output.shape)
    cv.imwrite('out.png', output)

test()
