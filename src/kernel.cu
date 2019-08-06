#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string.h>

__device__ __constant__ float D_H[ 3*3 ];

__device__ float norm(float val, int length) {
    float mean = length/2;
    float std = length/2;
    return (val-mean)/std;
}

__device__ float unorm(float val, int length) {
    float mean = length/2;
    float std = length/2;
    return val*std + mean;
}

__device__ void projectedCoord(int x, int y, int *xp, int *yp, int xlen, int ylen) {

    //printf("%d, %d \n", x, y);
    //NORMALIZE INPUT
    float nx = norm(x,xlen);
    float ny = norm(y,ylen);

    //printf("%f, %f \n", nx, ny);
    int sH = 3;
    float w = 1; //Assume that the projection starts from y=1


    float hx = nx*D_H[ sH*0+0] + ny*D_H[ sH*0+1 ] + w*D_H[ sH*0+2 ];
    float hy = nx*D_H[ sH*1+0] + ny*D_H[ sH*1+1 ] + w*D_H[ sH*1+2 ];
    float hw = nx*D_H[ sH*2+0] + ny*D_H[ sH*2+1 ] + w*D_H[ sH*2+2 ];

    //printf("%f, %f, %f\n",D_H[ sH*0+0], D_H[ sH*0+1], D_H[ sH*0+2]);
    //printf("%f, %f, %f\n",D_H[ sH*1+0], D_H[ sH*1+1], D_H[ sH*1+2]);
    //printf("%f, %f, %f\n",D_H[ sH*2+0], D_H[ sH*2+1], D_H[ sH*2+2]);
    //printf("%f %f %f \n", hx, hy, hw);

    //Unormalize Output
    *xp = unorm(hx/hw, xlen);
    *yp = unorm(hy/hw, ylen);

    //printf("%d, %d \n", *xp, *yp);

}

__device__ int im_idx(int r, int c, int width, int channels) {
    return channels*(width*r+c);
}

__device__ bool val_rc(int r, int c, int width, int height) {
    return r>=0 && r<height && c>=0 && c<width;
}

__global__ void proj_sub_tresh(unsigned char* img0, unsigned char* img1, unsigned char* out_img, int Width, int Height) {
    const unsigned int c = ( (blockDim.y * blockIdx.y) + threadIdx.y );
    const unsigned int r = ( (blockDim.x * blockIdx.x) + threadIdx.x );

    const unsigned int treshold = 60;
    const unsigned int ch = 3; //Channel
    const unsigned int s = sizeof(unsigned char);
    const unsigned int W = Width;
    int o_img_idx;
    int i_img_idx;
    unsigned int subval, subval0, subval1, subval2;
    int rp;
    int cp;
    //Projection, Background Sub, Treshold

    // Not sure why I wrote the matrix in this manner where the r column is reversed using x,y notation
    // Need to look deeper into and be fixed
    projectedCoord(c,r,  &cp, &rp, Width, Height);

    //printf("%d, %d \n", rp, cp);
    if ( val_rc(rp,cp, Width, Height) && val_rc(r,c, Width, Height) ) {


        o_img_idx = im_idx(r,c,  Width,ch);
        i_img_idx = im_idx(rp,cp,  Width,ch);
        subval0 = abs( img1[ o_img_idx+0 ] - img0[ i_img_idx+0 ] );
        subval1 = abs( img1[ o_img_idx+1 ] - img0[ i_img_idx+1 ] );
        subval2 = abs( img1[ o_img_idx+2 ] - img0[ i_img_idx+2 ] );
        subval = .21265*subval0 + .7152*subval1 + .0722*subval2;

        if (subval > treshold) {
            out_img[ o_img_idx+0 ] = subval;
            out_img[ o_img_idx+1 ] = subval;
            out_img[ o_img_idx+2 ] = subval;
        }
        //out_img[ o_img_idx+0 ] = img0[ i_img_idx+0];
        //out_img[ o_img_idx+1 ] = img0[ i_img_idx+1];
        //out_img[ o_img_idx+2 ] = img0[ i_img_idx+2];


    }
}
