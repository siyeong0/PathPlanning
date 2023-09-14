import numpy as np
import cv2 as cv
from utils import generate_random_map
from scanner import *

def test_scanner_control():
    world = generate_random_map(options=["discrete", "straight"])
    scanner = Scanner(world, 120, 48)

    terminate = False
    while not terminate:
        buffer = np.full((world.shape[0], world.shape[1], 3), 0, dtype=np.uint8)      # Clear
        scan_map, _ = scanner.scan()

        buffer[np.where(scan_map == UNKOWN)] = (0,0,0)
        buffer[np.where(scan_map == EMPTY)] = (64,64,64)
        buffer[np.where(scan_map == SCANNED)] = (255,0,255)
        buffer[np.where(scan_map == IN_VIEW)] = (0,128,128)
        buffer = cv.circle(buffer, (int(scanner.position[1]), int(scanner.position[0])), 5, (255,0,0), cv.FILLED)

        buffer = np.swapaxes(buffer, 0, 1)
        cv.imshow("Test scanner control", cv.resize(buffer, (512,512)))

        key = cv.waitKey(16)

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
