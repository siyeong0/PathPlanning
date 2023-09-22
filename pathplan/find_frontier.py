import numpy as np
import cv2 as cv
from scanner.scanner import UNKOWN, EMPTY, IN_VIEW, SCANNED

def find_frontier(map:np.ndarray):
    shape = map.shape
    buffer = np.zeros_like(map, dtype=np.uint8)
    buffer[np.where(map==EMPTY)] = 255
    buffer[np.where(map==IN_VIEW)] = 255
    contours, _ = cv.findContours(buffer, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    frontier_points = []
    for cb in list(contours):
        for cbb in cb:
            for c in cbb:
                cy, cx = c
                if cx == 0 or cx == shape[0] - 1 or cy == 0 or cy == shape[1] - 1:
                    continue
                if map[cx+1,cy] == UNKOWN or map[cx-1,cy] == UNKOWN or map[cx,cy+1] == UNKOWN or map[cx,cy-1] == UNKOWN:
                    frontier_points.append(np.array([cx, cy]))

    return frontier_points

def draw_frontiers(image:np.ndarray, frontier_points:list, color=(192,192,32)):
    for p in frontier_points:
        x, y = p
        image[x,y] = color