cimport cython
import numpy as np
cimport numpy as np

cdef extern from "src/manager.hh":
    void mov_obj_detect( unsigned char* , unsigned char* , unsigned char* , float* , int , int )

def c_mov_obj_detect(img0, img1, H_filter, width, height):
    cdef np.ndarray[unsigned char, ndim=1, mode="c"] c_img0 = img0.astype(np.uint8) 
    cdef np.ndarray[unsigned char, ndim=1, mode="c"] c_img1 = img1.astype(np.uint8) 
    cdef np.ndarray[float, ndim=1, mode="c"] c_H = H_filter.astype(np.float32)
    cdef np.ndarray[unsigned char, ndim=1, mode="c"] c_out_img = np.zeros( width*height*3, dtype=np.uint8)

    mov_obj_detect(&c_img0[0], &c_img1[0], &c_out_img[0], &c_H[0], width, height)

    return c_out_img
