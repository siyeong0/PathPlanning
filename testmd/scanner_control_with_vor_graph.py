
import numpy as np
import cv2 as cv
from utils import generate_random_map
from scanner import *
from pathplan import Graph, to_voronoi, extract_contour_graph

def test_scanner_control_with_vor_graph():
    world = generate_random_map(options=["discrete", "straight", "border"])
    scanner = Scanner(world, 120, 48)

    terminate = False
    voronoi = Graph([], [])
    target_indices = []
    buffer = np.full((world.shape[0], world.shape[1], 3), 0, dtype=np.uint8)      # Clear
    while not terminate:
        # Do scan
        scan_map, _ = scanner.scan()

        # Make graph when graph dirty
        gray_img = np.zeros_like(scan_map, dtype=np.uint8)
        gray_img[np.where(scan_map==SCANNED)] = 255
        contours = extract_contour_graph(gray_img, True)

        valid_map = np.full(scan_map.shape, False, dtype=np.bool_)
        valid_map[np.where(scan_map==EMPTY)] = True
        valid_map[np.where(scan_map==IN_VIEW)] = True

        if len(contours.vertices) > 4:
            voronoi, target_indices = to_voronoi(contours, valid_map)

        # Draw scan_map
        buffer[np.where(scan_map == UNKOWN)] = (0,0,0)
        buffer[np.where(scan_map == EMPTY)] = (64,64,64)
        buffer[np.where(scan_map == SCANNED)] = (255,0,255)
        buffer[np.where(scan_map == IN_VIEW)] = (0,128,128)

        # Draw graph
        buffer = voronoi.draw(buffer)
        for i in target_indices:    # Draw target vertices
            v = voronoi.vertices[i]
            v = (int(v[0]), int(v[1]))
            buffer = cv.rectangle(buffer, (v[1]-2, v[0]-2), (v[1]+2, v[0]+2), (0,0,255), cv.FILLED)
        # Draw scanner
        buffer = cv.circle(buffer, (int(scanner.position[1]), int(scanner.position[0])), 5, (255,0,0), cv.FILLED)

        # Render
        cv.imshow("Test scanner control", cv.resize(np.swapaxes(buffer, 0, 1), (512,512)))
        key = cv.waitKey(100)

        speed = 3.0
        if key==ord('w'):
            scanner.moveUp(speed)
        elif key==ord('s'):
            scanner.moveDown(speed)
        elif key==ord('a'):
            scanner.moveLeft(speed)
        elif key==ord('d'):
            scanner.moveRight(speed)
        elif key==ord('q'):
            scanner.rotate(30)
        elif key==ord('e'):
            scanner.rotate(-30)

        elif key==27:     # ESC
            terminate = True
        elif key==ord(' '): # SPACE
            cv.waitKey()

    cv.destroyAllWindows()