/*
This is the central piece of code. This file implements a class
 that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string.h>

using namespace std;

int img_s(int width, int height, int chan, int size) {
    return width*height*chan*size;
}

void mov_obj_detect( unsigned char* img0, unsigned char* img1, unsigned char* out_img, float* H_filter, int W, int H) {
    cudaMemcpyToSymbol( D_H, H_filter, 3*3*sizeof(float) );
    // Alloc Cpy Var
    unsigned int s = sizeof(unsigned char);
    unsigned int c = 3;

    unsigned char* D_img0;
    cudaMalloc( (void **) &D_img0, img_s(W,H,c,s) );
    cudaMemcpy( D_img0, img0, img_s(W,H,c,s), cudaMemcpyHostToDevice);

    unsigned char* D_img1;
    cudaMalloc( (void **) &D_img1, img_s(W,H,c,s) );
    cudaMemcpy( D_img1, img1, img_s(W,H,c,s), cudaMemcpyHostToDevice);

    unsigned char* D_out_img;
    cudaMalloc( (void **) &D_out_img, img_s(W,H,c,s));

    // Call Kernel
    unsigned int n = 32; // Block_Size
    unsigned int N = ceil( ((double) H) /n); // Grid Rows
    unsigned int M = ceil( ((double) W)/n); // Grid Cols

    dim3 gridDims(N,M,1);
    dim3 blockDims(n,n,1);
    proj_sub_tresh<<< gridDims, blockDims >>>(D_img0, D_img1, D_out_img, W, H);
    cudaMemcpy( out_img, D_out_img, img_s(W,H,c,s), cudaMemcpyDeviceToHost);

    // Free Var
    cudaFree(D_img0);
    cudaFree(D_img1);
    cudaFree(D_out_img);
}


