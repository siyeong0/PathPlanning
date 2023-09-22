import cv2 as cv
import numpy as np
from utils.generate_map import generate_random_map

def test_map_gen():
    map = generate_random_map()
    buffer = np.zeros(map.shape, dtype=np.uint8)
    buffer[np.where(map == 255)] = 0
    buffer[np.where(map == 0)] = 255
    
    cv.imshow("test", map)
    cv.waitKey()
    cv.destroyAllWindows()