# -*- coding: utf-8 -*-

import sys
import cv2
import os
import numpy as np

FILE_ROOT = os.getcwd()


def main():
    radius = 8
    offset = 0  # 0~63
    original_img = cv2.imread('./lightfield-images/lfc-dragons.telephoto-1200.tiff')

    size = 150, 150, 3
    m = np.zeros(size, dtype=np.uint8)

    while offset < 64:
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                offsetX = offset % radius
                offsetY = offset / radius
                m[y][x] = original_img[radius * y + offsetX][radius * x + offsetY]
        cv2.imwrite(os.path.join('refocused', 'refocus-' + `offsetX` + '_' + `offsetY` + '.tiff'), m)
        offset += 1


if __name__ == '__main__':
    main()
