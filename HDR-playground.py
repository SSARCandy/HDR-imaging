# coding: utf-8

import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cProfile
import constant


def load_exposures(source_dir, channel=0):
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]
    
    img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
    img_list = [img[:,:,channel] for img in img_list]
    exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times)


# In[3]:

def hdr_debvec(img_list, exposure_times):
    B = [math.log(e,2) for e in exposure_times]
    l = constant.L
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]

    small_img = [cv2.resize(img, (10, 10)) for img in img_list]
    Z = [img.flatten() for img in small_img]

    
    print(np.shape(Z))
    
    return response_curve_solver(Z, B, l, w)


# In[4]:

def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0)*np.size(Z, 1)+n+1, n+np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = Z[j][i]
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij*B[j]
            k += 1
    
    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


# In[5]:

def construct_radiance_map(g, Z, ln_t, w):
    acc_E = [0]*len(Z[0])
    ln_E = [0]*len(Z[0])
    
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z]*(g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else acc_E[i]
        acc_w = 0
    
    return ln_E

def construct_hdr(img_list, response_curve, exposure_times):
    # Construct radiance map for each channels
    img_size = img_list[0][0].shape
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]
    ln_t = np.log2(exposure_times)

    vfunc = np.vectorize(lambda x:math.exp(x))
    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

    # construct radiance map for BGR channels
    for i in range(3):
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(vfunc(E), img_size)

    return hdr


# Code based on https://gist.github.com/edouardp/3089602
def save_hdr(hdr, filename):
    image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
    image[..., 0] = hdr[..., 2]
    image[..., 1] = hdr[..., 1]
    image[..., 2] = hdr[..., 0]

    f = open(filename, 'wb')
    f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1]) 
    f.write(bytes(header, encoding='utf-8'))

    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('[Usage] python script <input img dir> <output .hdr name>')
        print('[Exampe] python script taipei taipei.hdr')
        sys.exit(0)
 
    img_dir, output_hdr_filename = sys.argv[1], sys.argv[2]

    # Loading exposure images into a list
    img_list_b, exposure_times = load_exposures(img_dir, 0)
    img_list_g, exposure_times = load_exposures(img_dir, 1)
    img_list_r, exposure_times = load_exposures(img_dir, 2)

    gb, _ = hdr_debvec(img_list_b, exposure_times)
    gg, _ = hdr_debvec(img_list_g, exposure_times)
    gr, _ = hdr_debvec(img_list_r, exposure_times)


    # Show response curve
    plt.figure(figsize=(10, 10))
    plt.plot(gr, range(256), 'r')
    plt.plot(gg, range(256), 'g')
    plt.plot(gb, range(256), 'b')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.show()

    hdr = construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)

    # Display Radiance map with pseudo-color image (log value)
    plt.figure(figsize=(12,8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)))
    plt.colorbar()
    plt.show()

    save_hdr(hdr, output_hdr_filename)
