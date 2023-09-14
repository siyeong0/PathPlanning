import numpy as np
import cv2 as cv
from utils import generate_random_map
from pathplan import Graph, to_voronoi, extract_contour_graph

def test_vor_graph_generation():
    random_map = generate_random_map(options=["discrete", "straight"])
    contours = extract_contour_graph(random_map)

    valid_map = np.full(random_map.shape, True, dtype=np.bool_)
    valid_map[np.where(random_map!=0)] = False
    voronoi, target_indices = to_voronoi(contours, valid_map)

    # Render
    buffer_shape = (random_map.shape[0], random_map.shape[1], 3)
    buffer = np.full(buffer_shape, 255, dtype=np.uint8) # Clear to white
    # Draw obstacles
    buffer[np.where(random_map==255)] = (0,0,0)
    # Draw contours
    for edge in contours.edges:
        v1, v2 = contours.vertices[edge[0]], contours.vertices[edge[1]]
        buffer = cv.line(buffer, (int(v1[1]), int(v1[0])), (int(v2[1]), int(v2[0])), (255,0,255), 1)
    # Draw edges
    for edge in voronoi.edges:
        v1, v2 = voronoi.vertices[edge[0]], voronoi.vertices[edge[1]]
        buffer = cv.line(buffer, (int(v1[1]), int(v1[0])), (int(v2[1]), int(v2[0])), (0,255,0), 1)
    # Draw vertices
    for v in voronoi.vertices:
        buffer = cv.circle(buffer, (int(v[1]), int(v[0])), 3, (255,0,0), cv.FILLED)
    # Draw target vertices
    for i in target_indices:
        v = voronoi.vertices[i]
        buffer = cv.circle(buffer, (int(v[1]), int(v[0])), 3, (0,0,255), cv.FILLED)
    
    # Present
    buffer = np.swapaxes(buffer, 0, 1)
    cv.imshow("Test voronoi graph generation", cv.resize(buffer,(512,512)))
    cv.waitKey()

