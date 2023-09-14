import gym
from gym import spaces
import numpy as np
import cv2 as cv

from utils import generate_random_map
from scanner import *
from pathplan import Graph, to_voronoi, extract_contour_graph

class EnvWrapper(gym.Env):
    def __init__(self, 
                 world_shape=(256,256),
                 obs_shape=(128,128),
                 speed = 2.0,
                 ang_speed = 15,
                 mpa_gen_options = ["discrete", "straight", "border"],
                 display_shape = (1024,1024)
                 ):
        super(EnvWrapper, self).__init__()
        self.world_shape = world_shape
        self.obs_shape = obs_shape
        self.speed = speed
        self.ang_speed = ang_speed
        self.options = mpa_gen_options
        self.display_shape = display_shape
    
        self.world = None
        self.scanner = None

        self.observation_space = spaces.Box(0, 255, self.obs_shape, dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete([5,3])

    def reset(self, seed=None, options=None):
        self.world = generate_random_map(shape=self.world_shape, options=self.options)
        self.scanner = Scanner(self.world, 120, 48)

        return self._get_obs(), {}
    
    def step(self, action):
        tr_action, rot_action = action
        is_valid_action = True
        # Translation
        if tr_action == 0:
            pass
        elif tr_action == 1:
            is_valid_action = self.scanner.moveUp(self.speed)
        elif tr_action == 2:
            is_valid_action = self.scanner.moveDown(self.speed)
        elif tr_action == 3:
            is_valid_action = self.scanner.moveLeft(self.speed)
        elif tr_action == 4:
            is_valid_action = self.scanner.moveRight(self.speed)
        else:
            assert(False)
        # Rotation
        if rot_action == 0:
            pass
        elif rot_action == 1:
            self.scanner.rotate(self.ang_speed)
        elif rot_action == 2:
            self.scanner.rotate(-self.ang_speed)
        else:
            assert(False)

        reward = 0.0
        done = False

        return self._get_obs(), reward, done, False, {}

#-----------------------------------------------------------------------------------------------
# Private
#-----------------------------------------------------------------------------------------------
    def _get_obs(self):
        curr_pos = self.scanner.position
        left_x, top_y = (np.array(curr_pos) - np.array(self.obs_shape)).astype(int)
        offset_x = [0 if left_x >= 0 else -left_x, left_x if left_x < self.obs_shape[0] else left_x - (self.obs_shape[0] - 1)]
        offset_y = [0 if top_y >= 0 else -top_y, top_y if top_y < self.obs_shape[1] else top_y - (self.obs_shape[1] - 1)]

        scan_map, _ = self.scanner.scan()
        obs = np.full(self.obs_shape, UNKOWN, dtype=np.uint8)
        obs[offset_x[0]:offset_x[1], offset_y[0]:offset_y[1]] = scan_map[left_x:left_x+self.obs_shape[0], top_y:top_y+self.obs_shape[1]]
        obs[self.scanner.position[0], self.scanner.position[1]] = AGENT

        return obs

#-----------------------------------------------------------------------------------------------
# Rendering
#-----------------------------------------------------------------------------------------------
    def render(self):
        world_buffer = np.full((self.world_shape[0], self.world_shape[1], 3), 0, dtype=np.uint8)      # Clear
        obs_buffer = np.full((self.obs_shape[0], self.obs_shape[1], 3), 0, dtype=np.uint8)      # Clear
        
        # Render scan map
        scan_map = self.scanner.scan_map
        curr_pos = self.scanner.position
        world_buffer[np.where(scan_map == UNKOWN)] = (0,0,0)
        world_buffer[np.where(scan_map == EMPTY)] = (64,64,64)
        world_buffer[np.where(scan_map == SCANNED)] = (255,0,255)
        world_buffer[np.where(scan_map == IN_VIEW)] = (0,128,128)
        world_buffer = cv.circle(world_buffer, (int(curr_pos[1]), int(curr_pos[0])), 5, (255,0,0), cv.FILLED)
        world_buffer = np.swapaxes(world_buffer, 0, 1)

        # Render observation
        obs = self._get_obs()
        obs_buffer[np.where(obs == UNKOWN)] = (0,0,0)
        obs_buffer[np.where(obs == EMPTY)] = (64,64,64)
        obs_buffer[np.where(obs == SCANNED)] = (255,0,255)
        obs_buffer[np.where(obs == IN_VIEW)] = (0,128,128)
        obs_buffer = cv.circle(obs_buffer, (int(self.scanner.position[0]), int(self.scanner.position[1])), 3, (255,0,0), cv.FILLED)
        obs_buffer = np.swapaxes(obs_buffer, 0, 1)

        obs_buffer = cv.resize(obs_buffer, self.display_shape)
        minimap_shape = (int(self.display_shape[0] * 0.2), int(self.display_shape[1] * 0.2))
        world_buffer = cv.resize(world_buffer, minimap_shape)
        obs_buffer[self.display_shape[0] - minimap_shape[0]:0,0:minimap_shape[1]]

        cv.imshow("Test scanner control", cv.resize(obs_buffer, (512,512)))
        key = cv.waitKey(16)

        if key==27:     # ESC
            terminate = True
        elif key==ord(' '): # SPACE
            cv.waitKey()