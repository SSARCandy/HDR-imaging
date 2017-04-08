# coding: utf-8

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cProfile


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
    l = 50
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

    print(np.shape(A))
    print(np.shape(b))
    
    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


# In[5]:


# In[7]:

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
        ln_E[i] = acc_E[i]/acc_w
        acc_w = 0
    
    return ln_E


# In[8]:

# Code borrowed from https://gist.github.com/edouardp/3089602
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
    # Loading exposure images into a list
    img_list_b, exposure_times = load_exposures('test', 0)
    img_list_g, exposure_times = load_exposures('test', 1)
    img_list_r, exposure_times = load_exposures('test', 2)

    gb, _ = hdr_debvec(img_list_b, exposure_times)
    gg, _ = hdr_debvec(img_list_g, exposure_times)
    gr, _ = hdr_debvec(img_list_r, exposure_times)


    # Show response curve
    plt.figure(figsize=(10,10))
    plt.plot(gr,range(256), 'r')
    plt.plot(gg,range(256), 'g')
    plt.plot(gb,range(256), 'b')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.show()


    # Construct radiance map for each channels
    img_size = img_list_b[0].shape
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]
    Zb = [img.flatten().tolist() for img in img_list_b]
    Zg = [img.flatten().tolist() for img in img_list_g]
    Zr = [img.flatten().tolist() for img in img_list_r]

    cProfile.run('Eb = construct_radiance_map(gb, Zb, exposure_times, w)')
    cProfile.run('Eg = construct_radiance_map(gg, Zg, exposure_times, w)')
    cProfile.run('Er = construct_radiance_map(gr, Zr, exposure_times, w)')

    E = np.asarray([Eb, Eg, Er])

    # Exponational each channels
    vfunc = np.vectorize(lambda x:math.exp(x))
    for i in range(3):
        E[i] = vfunc(E[i])
    
    bE = np.reshape(E[0], img_size)
    gE = np.reshape(E[1], img_size)
    rE = np.reshape(E[2], img_size)

    # Merge RGB to one matrix
    hdr = np.zeros((rE.shape[0], rE.shape[1], 3), 'float32')
    hdr[..., 0] = bE
    hdr[..., 1] = gE
    hdr[..., 2] = rE

    # Display Radiance map with pseudo-color image (log value)
    uhdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(12,8))
    plt.imshow(np.log(uhdr))
    plt.colorbar()
    plt.show()

    save_hdr(hdr, 'test.hdr')
