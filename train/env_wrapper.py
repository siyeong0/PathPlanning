import gym
from gym import spaces
import numpy as np
import cv2 as cv

from utils import generate_random_map
from scanner import *
from pathplan import Graph, extract_contour_graph
from pathplan.find_frontier import find_frontier, draw_frontiers

CLEAR_REWARD = 100.0
INVALID_REWARD = -10.0

class Scanning_a6(gym.Env):
    def __init__(self, 
                 obs_shape=(96,96),
                 speed = 3.0,
                 ang_speed = 30,
                 display_shape = (1024,1024),
                 ):
        super(Scanning_a6, self).__init__()
        self.obs_shape = obs_shape
        self.speed = speed
        self.ang_speed = ang_speed
        self.display_shape = display_shape
    
        self.world = None
        self.scanner = None
        self.prev_obs = None
        self.total_ep_step = None

        self.observation_space = spaces.Box(0, 255, (3, self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)

        self.ep_terminate = None
        self.clear_scanning = None

        self.render_mode = None

    def reset(self, seed=None, options=None):
        map_options = ["discrete"]# if np.random.randint(0,2) == 1 else ""]
        self.world = generate_random_map(shape=self.obs_shape, options=map_options)
        while len(list(np.where(self.world==0))) == 0:
            self.world = generate_random_map(shape=self.obs_shape, options=map_options)
        self.scanner = Scanner(self.world, 120, 48)

        self.curr_scanned = 0
        self.total_scanned = 0
        contours, _ = cv.findContours(self.world, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        self.max_scanned = 0
        for cb in list(contours):
            for i in range(1, cb.shape[0]):
                x, y = ((cb[i][0][1], cb[i][0][0]))
                if x == 0 or x == self.world.shape[0] - 1 or y == 0 or y == self.world.shape[1] - 1:
                    continue
                self.max_scanned += 1

        self.ep_terminate = False
        self.clear_scanning = False
        self.total_ep_step = 0

        return self._get_obs(), {}
    
    def step(self, action):
        self.total_ep_step += 1
        is_valid_action = True
        self.move = action >= 0 and action < 4

        if action == 0:
            is_valid_action = self.scanner.moveUp(self.speed)
        elif action == 1:
            is_valid_action = self.scanner.moveDown(self.speed)
        elif action == 2:
            is_valid_action = self.scanner.moveLeft(self.speed)
        elif action == 3:
            is_valid_action = self.scanner.moveRight(self.speed)
        elif action == 4:
            self.scanner.rotate(self.ang_speed)
        elif action == 5:
            self.scanner.rotate(-self.ang_speed)
        else:
            assert(False)
            
        obs = self._get_obs()
        reward = self._get_reward(is_valid_action, self.scan_valid)
        done = self.ep_terminate

        return obs, reward, done, False, {}

#-----------------------------------------------------------------------------------------------
# Private
#-----------------------------------------------------------------------------------------------
    def _get_obs(self):
        obs = np.full((self.obs_shape[1], self.obs_shape[0], 3), 0, dtype=np.uint8)      # Clear

        scan_map, self.curr_scanned, self.scan_valid = self.scanner.scan()
        self.total_scanned += self.curr_scanned
        obs[np.where(scan_map==UNKOWN)] = (0,0,0)
        obs[np.where(scan_map==EMPTY)] = (64,64,64)
        obs[np.where(scan_map==IN_VIEW)] = (0,128,128)
        obs[np.where(scan_map==SCANNED)] = (255,0,255)
        obs = cv.circle(obs, ((int(self.scanner.position[1])), int(self.scanner.position[0])), 3, (255,0,0), cv.FILLED)

        if self.total_scanned >= self.max_scanned * 0.95:
            self.clear_scanning = True

        obs = np.swapaxes(obs, 0, 2)
        self.prev_obs = obs
        return obs
    
    def _get_reward(self, is_valid:bool, scan_valid:bool):
        if self.total_ep_step > 100:
            self.ep_terminate = True
            return INVALID_REWARD
        
        if not is_valid:
            self.ep_terminate = True
            return INVALID_REWARD

        if self.clear_scanning:
            self.ep_terminate = True
            return CLEAR_REWARD
        
        reward = -np.exp(-self.curr_scanned) + 0.0 if scan_valid else -1.0
        return reward
            
#-----------------------------------------------------------------------------------------------
# Rendering
#-----------------------------------------------------------------------------------------------
    def render(self):
        obs_buffer = np.swapaxes(self.prev_obs, 0, 2)
        obs_buffer = np.swapaxes(obs_buffer, 0, 1)
        cv.imshow("Test scanner control", cv.resize(obs_buffer,self.display_shape))
        cv.waitKey(16)