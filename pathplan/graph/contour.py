import numpy as np
import cv2 as cv
from pathplan.graph import Graph

def extract_contour_graph(gray_img: np.ndarray, vor:bool=False) -> Graph:
    contours, _ = cv.findContours(gray_img, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)

    idx = 0
    con_vertices = []
    con_edges = []
    for cb in contours:
        con_vertices.append((cb[0][0][1], cb[0][0][0]))
        con_edges.append([idx, idx+cb.shape[0]-1])
        idx += 1
        for i in range(1, cb.shape[0]):
            con_vertices.append((cb[i][0][1], cb[i][0][0]))
            con_edges.append([idx, idx - 1])
            idx += 1

    if vor:
        w, h = gray_img.shape
        slice = 4
        sw, sh = (w-1)/slice, (h-1)/slice
        for i in range(slice):
            con_vertices.append([int(sw * i), 0])
            con_vertices.append([0, int(sh * (i + 1))])
            con_vertices.append([int(sw * (i + 1)), h - 1])
            con_vertices.append([w - 1, int(sh * (i + 1))])

    return Graph(con_vertices, con_edges)