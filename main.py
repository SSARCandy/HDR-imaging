# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np

def load_exposures(source_dir):
    # imgs = [f for f in os.listdir(source_dir) if re.search(r'\.png$', f)]
    # print(imgs)

    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]
    
    img_list = [cv2.imread(os.path.join(source_dir, f)) for f in filenames]
    exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times)


def hdr(source_dir, output_name='hdr.jpg'):
    # Loading exposure images into a list
    img_list, exposure_times = load_exposures(source_dir)

    # Merge exposures to HDR image
    merge_debvec = cv2.createMergeDebevec()
    hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
    merge_robertson = cv2.createMergeRobertson()
    hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

    # Tonemap HDR image
    tonemap1 = cv2.createTonemapDurand(gamma=2.2)
    res_debvec = tonemap1.process(hdr_debvec.copy())
    tonemap2 = cv2.createTonemapDurand(gamma=1.3)
    res_robertson = tonemap2.process(hdr_robertson.copy())

    # Exposure fusion using Mertens
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)

    # Convert datatype to 8-bit and save
    res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
    res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

    cv2.imwrite("ldr_debvec.jpg", res_debvec_8bit)
    cv2.imwrite("ldr_robertson.jpg", res_robertson_8bit)
    cv2.imwrite("fusion_mertens.jpg", res_mertens_8bit)

if __name__ == '__main__':
    hdr('test', '')
