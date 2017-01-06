# -*- coding: utf-8 -*-

import sys
import cv2
import os
import numpy as np

def main(radius, file_path):
    offset = 0  # 0~63
    original_img = cv2.imread(file_path)

    size = 150, 150, 3
    m = np.zeros(size, dtype=np.uint8)

    # cv2.imshow('gt', original_img)
    # cv2.waitKey(0)
    while offset < radius**2:
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                offsetX = offset % radius
                offsetY = offset / radius
                m[y][x] = original_img[radius * y + offsetX][radius * x + offsetY]

        save_file_path = os.path.join('refocused', 'refocus-' + `offsetX` + '_' + `offsetY` + '.tiff')
        cv2.imwrite(save_file_path, m)
        print('Saving file ' + save_file_path + ' done.')
        offset += 1


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[Usage] python main.py <file_path> <lenslet_radius>')
        sys.exit(0)

    file_path = sys.argv[1]
    radius = int(sys.argv[2])
    main(radius, file_path)

# python main.py .\lightfield-images\lfc-dragons.dgauss-1200.tiff 8
# python main.py .\lightfield-images\lfc-dragons.telephoto-1200.tiff 8