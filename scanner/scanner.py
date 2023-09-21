import numpy as np

UNKOWN = 0
EMPTY = 1
SCANNED = 2
IN_VIEW = 3
AGENT = 4

class Scanner:
    def __init__(self, 
                 target_world: np.ndarray,
                 FOV = 120, max_depth = 48):
        self.world = None
        self.position = None
        self.yaw = None # Radian
        self.fov = None
        self.max_depth = None
        self.scan_map = None
        self.initialize(target_world, FOV, max_depth)

    def initialize(self, 
                   target_world: np.ndarray,
                   FOV = 120, max_depth = 48):
        self.world = target_world
        while True:
            self.position = np.array([float(np.random.randint(0,self.world.shape[0])), 
                                      float(np.random.randint(0,self.world.shape[1]))])
            if self.world[int(self.position[0]), int(self.position[1])] == 0:
                break
        self.yaw = 0.0 
        self.fov = FOV /180 * np.pi
        self.max_depth = max_depth
        self.scan_map = np.full(self.world.shape, UNKOWN, dtype=np.uint8)

    def move(self, dir, dist) -> bool:
        new_pos = self.position + dir * dist
        valid = self.world[int(new_pos[0]), int(new_pos[1])] == 0
        self.position = new_pos if valid else self.position
        return valid
    def moveUp(self, dist) -> bool:
        return self.move(np.array([0.0,-1.0]), dist)
    def moveDown(self, dist) -> bool:
        return self.move(np.array([0.0,1.0]), dist)
    def moveLeft(self, dist) -> bool:
        return self.move(np.array([-1.0,0.0]), dist)
    def moveRight(self, dist) -> bool:
        return self.move(np.array([1.0,0.0]), dist)

    def rotate(self, deg):
        self.yaw += deg * np.pi / 180
        self.yaw %= 2 * np.pi
    def lookTo(self, deg) -> bool:
        yaw = deg * np.pi / 180
        if yaw == self.yaw:
            return False
        self.yaw = deg * np.pi / 180
        self.yaw %= 2 * np.pi
        return True

    def scan(self) -> np.ndarray:
        self.scan_map[np.where(self.scan_map==IN_VIEW)] = EMPTY
        self.scan_map[np.where(self.scan_map==AGENT)] = EMPTY

        num_occupied_pixels = 0
        pos_x, pos_y = self.position[0], self.position[1]

        n_slice = 128
        fd = self.fov / n_slice
        for i in range(n_slice):
            dir = self.yaw - self.fov / 2 + fd * i
            curr_x = pos_x + np.cos(dir)
            curr_y = pos_y + np.sin(dir)
            for _ in range(int(self.max_depth)):
                x, y = int(curr_x), int(curr_y)
                if not self._is_valid_pos((x,y)):
                    break
                if self.world[x, y] != 0 :  # Collide with obstacle
                    if self.scan_map[x,y] != SCANNED:
                        self.scan_map[x,y] = SCANNED
                        num_occupied_pixels += 1
                    break
                else:
                    self.scan_map[x, y] = IN_VIEW
                curr_x += np.cos(dir)
                curr_y += + np.sin(dir)
        
        self.scan_map[int(self.position[0]), int(self.position[1])] = AGENT

        return self.scan_map.copy(), num_occupied_pixels

    def setFOV(self, deg):
        self.fov = deg / np.pi

    def _is_valid_pos(self, pos):
        w, h = self.world.shape
        x,y = int(pos[0]), int(pos[1])
        return 0 <= x and x < w and 0 <= y and y < h
        
