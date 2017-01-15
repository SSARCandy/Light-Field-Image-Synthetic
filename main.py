# -*- coding: utf-8 -*-

import sys
import cv2
import os
import math
import numpy as np
import progressbar
import multiprocessing
from joblib import Parallel, delayed

def integral_pixel(original_img, middle_x, middle_y, radius, shift=1):
    integral = [0, 0, 0]

    for y in range(-radius/2, radius/2, 1):
        for x in range(-radius/2, radius/2, 1):
            xx = middle_x + int(x*shift)
            yy = middle_y + int(y*shift)
            if (xx < 0 or yy < 0 or xx >= original_img.shape[1] or yy >= original_img.shape[0]):
                continue
            integral[0] += original_img[yy][xx][0]
            integral[1] += original_img[yy][xx][1]
            integral[2] += original_img[yy][xx][2]
    
    return integral
    

def extract_subaperture_images(radius, file_path):
    offset = 0  # 0~63
    original_img = cv2.imread(file_path)

    size = original_img.shape[0]/radius, original_img.shape[1]/radius, 3
    m = np.zeros(size, dtype=np.uint8)

    while offset < radius**2:
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                offsetX = offset % radius
                offsetY = offset / radius
                m[y][x] = original_img[radius * y + offsetY][radius * x + offsetX]

        save_file_path = os.path.join('refocused', 'subaperture-' + `offsetY` + '_' + `offsetX` + '.png')
        cv2.imwrite(save_file_path, m)
        print('Saving file to "' + save_file_path + '" done.')
        offset += 1


def combine_subapertures(radius, file_path, alpha=0):
    original_img = cv2.imread(file_path)

    size = original_img.shape[0]/radius, original_img.shape[1]/radius, 3
    m = np.zeros(size, dtype=np.uint8)


    for a in range(-10,11,1):
        bar = progressbar.ProgressBar(maxval=m.shape[0], widgets=[progressbar.Bar('+', 'Syntheticing: [', ']'), ' ', progressbar.ETA()])
        bar.start()
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                pixel_sum = [0, 0, 0]
                pixel_sum = integral_pixel(original_img, radius * x + radius/2, radius * y + radius/2, radius, float(a/5.0))
                m[y][x] = map(lambda x: x/radius**2, pixel_sum)
            
            bar.update(y+1)

        save_file_path = os.path.join('refocused', 'combine_'+`float(a/5.0)`+'.png')
        cv2.imwrite(save_file_path, m)
        bar.finish()
        print('Saving file to "' + save_file_path + '" done.')

def load_subapertures_from_dir(dir_path, radius):
    imgs = os.listdir(dir_path)
    
    subapertures = []
    tmp = []
    for i in range(len(imgs)):
        if i % radius == 0 and i > 0:
            subapertures.append(tmp)
            tmp = []
        sub_img = cv2.imread(os.path.join(dir_path, imgs[i]))
        sub_img = cv2.normalize(sub_img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        tmp.append(sub_img)
    subapertures.append(tmp)

    # for i in range(len(subapertures)):
    #     for j in range(len(subapertures[i])):
    #         print((i, j))
    #         cv2.imshow('y', subapertures[i][j])
    #         cv2.waitKey(0)

    return subapertures
  

def combine_subapertures2(dir_path='./refocused/dgauss-subapertures/', shift=0, radius=8):
    subapertures = load_subapertures_from_dir(dir_path)

    m = np.zeros(subapertures[0][0].shape, dtype=np.float32)
    for y in range(len(subapertures)):
        for x in range(len(subapertures[y])):
            padding = [0, 0, 0, 0] # top, bottom, left, right
            middle = (radius/2.0) if radius % 2 == 0 else (radius-1)/2.0
            if x - middle < 0:
                padding[2] = abs(shift*int(x - middle-0.5))
            if x - middle > 0:
                padding[3] = abs(shift*int(x - middle+0.5))
            if y - middle < 0:
                padding[0] = abs(shift*int(y - middle-0.5))
            if y - middle > 0:
                padding[1] = abs(shift*int(y - middle+0.5))

            if shift < 0:
                padding[1], padding[0] = padding[0], padding[1]
                padding[3], padding[2] = padding[2], padding[3]
            print(padding)

            sub_img = cv2.copyMakeBorder(subapertures[y][x], padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, (255, 255, 255))
            sub_img = sub_img[padding[1]:m.shape[0]+padding[1], padding[3]:m.shape[1]+padding[3]]

            cv2.imshow('hhhh', sub_img)
            cv2.waitKey(0)
            cv2.accumulateWeighted(sub_img, m, 1/float(radius**2))
    
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX, 0)
    filename = `shift` if shift < 0 else '+' + `shift`
    save_file_path = os.path.join('refocused',  filename + 'refocus.png')
    cv2.imwrite(save_file_path, m)
    print('Saving file to "' + save_file_path + '" done.')
    

            
def digital_refocus(dir_path, radius=8, alpha=1):
    subapertures = load_subapertures_from_dir(dir_path, radius)

    width = subapertures[0][0].shape[0]*radius
    lenses = width/radius
    m = np.zeros((lenses, lenses, 3), dtype=np.float32)

    # bar = progressbar.ProgressBar(maxval=radius**2, widgets=[progressbar.Bar('+', 'Syntheticing: [', ']'), ' ', progressbar.ETA()])
    # bar.start()

    for i in range(0, radius, 1):
        for j in range(1, radius+1, 1):
            u = (-(radius/2.0) + i) / (radius/2.0) + (1 / float(radius))
            v = (-(radius/2.0) + j) / (radius/2.0) + (1 / float(radius))
            shift_x = u * (1 - 1.0 / alpha)
            shift_y = v * (1 - 1.0 / alpha)

            for y in range(lenses):
                for x in range(lenses):
                    xToSample = int(x - shift_x * float(lenses) + 0.5)
                    yToSample = int(y - shift_y * float(lenses) + 0.5)
                    if xToSample < lenses and xToSample >= 0 and yToSample < lenses and yToSample >= 0:
                        m[y][x] += subapertures[i][j-1][yToSample][xToSample]

            print('alpha='+('%.4f' % alpha) + ': '+`int((i*radius+j)/float(radius**2)*100)` + '%')
    #         bar.update(i*radius+j)
    # bar.finish()


    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = m[i][j]/float(radius**2)
    
    # cv2.imshow('hhhh', m)
    # cv2.waitKey(0)
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX, 0)
    save_file_path = os.path.join('refocused',  ('%.4f' % alpha) + '_refocus.png')
    cv2.imwrite(save_file_path, m)
    print('Saving file to "' + save_file_path + '" done.')
    return m





if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[Usage] python main.py <file_path> <lenslet_radius>')
        sys.exit(0)

    file_path = sys.argv[1]
    radius = int(sys.argv[2])
    # combine_subapertures(radius, file_path)
    # extract_subaperture_images(radius, file_path)
    # for i in range(-4,5,1):
    #     combine_subapertures2('./refocused/rrr', i, radius)
    # combine_subapertures2('./refocused/rrr', -1, radius)

    num_cores = multiprocessing.cpu_count()
    print('Detect ' + `num_cores` + ' cores')

    alpha_arr = np.arange(0.96, 1.04, 0.0025)

    results = Parallel(n_jobs=num_cores, verbose=50)(delayed(digital_refocus)('./refocused/dgauss-subapertures', 8, i) for i in alpha_arr)



'''
python main.py .\lightfield-images\lfc-dgauss-1200-150.tiff 8
python main.py .\lightfield-images\lfc-dgauss-500-100.tiff 5
python main.py .\lightfield-images\lfc-dgauss-dragons-1200-150.tiff 8
python main.py .\lightfield-images\lfc-dgauss-dragons2-1200-150.tiff 8
python main.py .\lightfield-images\lfc-dgauss-dragons-1360-80.tiff 17
python main.py .\lightfield-images\lfc-telephoto-1200-150.tiff 8
python main.py .\lightfield-images\lfc-telephoto-3600-450.tiff 8
python main.py .\lightfield-images\lfc-dgauss-1200-400.tiff 3
'''