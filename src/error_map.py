from distutils.log import error
import cv2
import math
import numpy as np


def error_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    se = ((img1 - img2)**2)
    #se = abs(img1 - img2)
    return se


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



img1 = cv2.imread("/home/cgv/NeRF/data_in_paper/nerf.png", 0)
img2 = cv2.imread("/home/cgv/NeRF/data_in_paper/our_black.png", 0)
gt = cv2.imread("/home/cgv/NeRF/data_in_paper/gt.png", 0)

psnr = calculate_psnr(img1, gt)
psnr2 = calculate_psnr(img2, gt)
print(psnr)
print(psnr2)#
error_map = error_psnr(img1,gt)
error_map2 = error_psnr(img2,gt)
np_error_map = np.array(error_map)
np_error_map2 = np.array(error_map2)
m = np.mean(np_error_map)
std = np.std(np_error_map)


cv2.imwrite("/home/cgv/NeRF/data_in_paper/nerf_errormap_gray.png",np_error_map )
# cv2.imwrite("/home/cgv/NeRF/data_in_paper/our_errormap.png",np_error_map2 )

