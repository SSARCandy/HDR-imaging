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

# MTB implementation
def median_threshold_bitmap_alignment(img_list):
    median = [np.median(img) for img in img_list]
    binary_thres_img = [cv2.threshold(img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(len(img_list))]
    mask_img = [cv2.inRange(img_list[i], median[i]-20, median[i]+20) for i in range(len(img_list))]
 
    plt.imshow(mask_img[0], cmap='gray')
    plt.show()
    
    max_offset = np.max(img_list[0].shape)
    levels = 5

    global_offset = []
    for i in range(0, len(img_list)):
        offset = [[0,0]]
        for level in range(levels, -1, -1):
            scaled_img = cv2.resize(binary_thres_img[i], (0, 0), fx=1/(2**level), fy=1/(2**level))
            ground_img = cv2.resize(binary_thres_img[0], (0, 0), fx=1/(2**level), fy=1/(2**level))
            ground_mask = cv2.resize(mask_img[0], (0, 0), fx=1/(2**level), fy=1/(2**level))
            mask = cv2.resize(mask_img[i], (0, 0), fx=1/(2**level), fy=1/(2**level))
            
            level_offset = [0, 0]
            diff = float('Inf')
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    off = [offset[-1][0]*2+y, offset[-1][1]*2+x]
                    error = 0
                    for row in range(ground_img.shape[0]):
                        for col in range(ground_img.shape[1]):
                            if off[1]+col < 0 or off[0]+row < 0 or off[1]+col >= ground_img.shape[1] or off[0]+row >= ground_img.shape[1]:
                                continue
                            if ground_mask[row][col] == 255:
                                continue
                            error += 1 if ground_img[row][col] != scaled_img[y+off[0]][x+off[1]] else 0
                    if error < diff:
                        level_offset = off
                        diff = error
            offset += [level_offset]
        global_offset += [offset[-1]]
    return global_offset


def hdr_debvec(img_list, exposure_times, number_of_samples_per_dimension=20):
    B = [math.log(e,2) for e in exposure_times]
    l = constant.L
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]

    samples = []
    width = img_list[0].shape[0]
    height = img_list[0].shape[1]
    width_iteration = width / number_of_samples_per_dimension
    height_iteration = height / number_of_samples_per_dimension

    w_iter = 0
    h_iter = 0

    Z = np.zeros((len(img_list), number_of_samples_per_dimension*number_of_samples_per_dimension))
    for img_index, img in enumerate(img_list):
        h_iter = 0
        for i in range(number_of_samples_per_dimension):
            w_iter = 0
            for j in range(number_of_samples_per_dimension):
                if math.floor(w_iter) < width and math.floor(h_iter) < height:
                    pixel = img[math.floor(w_iter), math.floor(h_iter)]
                    Z[img_index, i * number_of_samples_per_dimension + j] = pixel
                w_iter += width_iteration
            h_iter += height_iteration
    
    return response_curve_solver(Z, B, l, w)


# Implementation of paper's Equation(3) with weight
def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0)*np.size(Z, 1)+n+1, n+np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = int(Z[j][i])
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

# Implementation of paper's Equation(6)
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
        print(' - Constructing radiance map for {0} channel .... '.format('BGR'[i]), end='', flush=True)
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(vfunc(E), img_size)
        print('done')

    return hdr

def hdr2ldr(hdr, filename):
    tonemap = cv2.createTonemapDrago(5)
    ldr = tonemap.process(hdr)
    cv2.imwrite('{}.png'.format(filename), ldr * 255)

# main
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('[Usage] python script <input img dir> <output .hdr name>')
        print('[Exampe] python script taipei taipei.hdr')
        sys.exit(0)
 
    img_dir, output_hdr_filename = sys.argv[1], sys.argv[2]

    # Loading exposure images into a list
    print('Reading input images.... ', end='')
    img_list_b, exposure_times = load_exposures(img_dir, 0)
    img_list_g, exposure_times = load_exposures(img_dir, 1)
    img_list_r, exposure_times = load_exposures(img_dir, 2)
    print('done')

    # Solving response curves
    print('Solving response curves .... ', end='')
    gb, _ = hdr_debvec(img_list_b, exposure_times)
    gg, _ = hdr_debvec(img_list_g, exposure_times)
    gr, _ = hdr_debvec(img_list_r, exposure_times)
    print('done')


    # Show response curve
    print('Saving response curves plot .... ', end='')
    plt.figure(figsize=(10, 10))
    plt.plot(gr, range(256), 'rx')
    plt.plot(gg, range(256), 'gx')
    plt.plot(gb, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig('response-curve.png')
    print('done')

    print('Constructing HDR image: ')
    hdr = construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)
    print('done')

    # Display Radiance map with pseudo-color image (log value)
    print('Saving pseudo-color radiance map .... ', end='')
    plt.figure(figsize=(12,8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('radiance-map.png')
    print('done')

    print('Saving HDR image .... ', end='')
    cv2.imwrite(output_hdr_filename, hdr)
    print('done')

    print('Saving LDR image .... ', end='')
    hdr2ldr(hdr, output_hdr_filename)
    print('done')
