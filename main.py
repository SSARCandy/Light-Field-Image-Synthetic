# -*- coding: utf-8 -*-

import sys
import cv2
import os
import numpy as np
import progressbar

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

        save_file_path = os.path.join('refocused', 'refocus-' + `offsetX` + '_' + `offsetY` + '.png')
        cv2.imwrite(save_file_path, m)
        print('Saving file ' + save_file_path + ' done.')
        offset += 1

def combine_subapertures(radius, file_path, alpha=0):
    original_img = cv2.imread(file_path)

    size = original_img.shape[0]/radius, original_img.shape[1]/radius, 3
    m = np.zeros(size, dtype=np.uint8)

    bar = progressbar.ProgressBar(maxval=m.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for y in range(m.shape[0]):
        for x in range(m.shape[1]):
            pixel_sum = [0, 0, 0]
            for offsetY in range(radius):
                for offsetX in range(radius):
                    pixel_sum[0] += original_img[radius * y + offsetY][radius * x + offsetX][0]
                    pixel_sum[1] += original_img[radius * y + offsetY][radius * x + offsetX][1]
                    pixel_sum[2] += original_img[radius * y + offsetY][radius * x + offsetX][2]
            pixel_sum[0] /= radius**2
            pixel_sum[1] /= radius**2
            pixel_sum[2] /= radius**2
            m[y][x] = pixel_sum
        
        bar.update(y+1)

    save_file_path = os.path.join('refocused', 'combine.png')
    cv2.imwrite(save_file_path, m)
    bar.finish()
    print('Saving file ' + save_file_path + ' done.')



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[Usage] python main.py <file_path> <lenslet_radius>')
        sys.exit(0)

    file_path = sys.argv[1]
    radius = int(sys.argv[2])
    combine_subapertures(radius, file_path)

# python main.py .\lightfield-images\lfc-dgauss-1200-150.tiff 8
# python main.py .\lightfield-images\lfc-telephoto-1200-150.tiff 8
# python main.py .\lightfield-images\lfc-telephoto-3600-450.tiff 8
# python main.py .\lightfield-images\lfc-dgauss-1200-400.tiff 3