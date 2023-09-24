import numpy as np
import cv2 as cv
from smath.perline_noise import generate_perlin_noise_2d

def to_straight(src, slice=None, val=1, border=True):
    dst = np.zeros_like(src, dtype=np.uint8)
    w, h = dst.shape
    if slice == None:
        slice = ((w+h)/2 / np.random.randint(16,32))

    stride_w = w / slice
    stride_h = h / slice
    for sw in range(int(slice)):
        for sh in range(int(slice)):
            rect = (int(sw * stride_w), int(sh * stride_h), int(stride_w), int(stride_h))
            sub_img = src[rect[0]:rect[0]+rect[2],rect[1]:rect[1]+rect[3]]
            num_total_pix = sub_img.shape[0] * sub_img.shape[1]
            num_wall_pix = (sub_img == val).sum()
            if num_wall_pix > num_total_pix * 0.6:
                dst[rect[0]:rect[0]+rect[2],
                    rect[1]:rect[1]+rect[3]] = val
                
    if border:
        dst[0:int(stride_w),:] = val
        dst[int(stride_w) * int(slice-1):dst.shape[0],:] = val
        dst[:,0:int(stride_h)] = val
        dst[:,int(stride_h) * int(slice-1):dst.shape[1]] = val
    return dst

def generate_random_map(shape=(256,256), options=["discrete","straight"]):
    noise = generate_perlin_noise_2d(shape, (1,1))
    map = noise.copy()
    if "discrete" in options:
        map = np.zeros(map.shape, dtype=np.uint8)
        map[np.where(noise>=0.15)] = 255
    if "straight" in options:
        map = to_straight(map, val=255, border="border" in options)
    return map