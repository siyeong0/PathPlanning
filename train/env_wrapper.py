import gym
from gym import spaces
import numpy as np
import cv2 as cv

from utils import generate_random_map
from scanner import *
from pathplan import Graph, to_voronoi, extract_contour_graph
from pathplan.find_frontier import find_frontier

INVALID_ACTION_REWARD = -10.0
POS_REWARD = 1.0
NEG_REWARD = -1.0

FRONTIER = AGENT + 1

class Scanning_a16(gym.Env):
    def __init__(self, 
                 obs_shape=(96,96),
                 speed = 4.0,
                 ang_speed = 15,
                 display_shape = (1024,1024),
                 ):
        super(Scanning_a16, self).__init__()
        self.obs_shape = obs_shape
        self.speed = speed
        self.ang_speed = ang_speed
        self.display_shape = display_shape
    
        self.world = None
        self.scanner = None

        self.num_classes = 5
        self.observation_space = spaces.Box(0.0, 1.0, (self.num_classes, self.obs_shape[0], self.obs_shape[1]), dtype=np.float32)
        self.action_space = spaces.Discrete(16)

        self.graph = None
        self.target_indices = None
        self.num_target_nodes = None
        self.ep_terminate = None
        self.clear_scanning = None
        self.num_prev_scanned = None
        
        self.render_mode = None

    def reset(self, seed=None, options=None):
        map_options = ["discrete", "straight" if np.random.randint(0,2) == 1 else ""]
        self.world = generate_random_map(shape=self.obs_shape, options=map_options)
        while len(list(np.where(self.world==0))) == 0:
            self.world = generate_random_map(shape=self.obs_shape, options=map_options)
        self.scanner = Scanner(self.world, 120, 48)

        self.graph = Graph([],[])
        self.target_indices = []
        self.num_target_nodes = 0
        self.ep_terminate = False
        self.clear_scanning = False
        self.num_prev_scanned = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        is_valid_action = True

        if action == 0:
            is_valid_action = self.scanner.moveUp(self.speed)
        elif action == 1:
            is_valid_action = self.scanner.moveDown(self.speed)
        elif action == 2:
            is_valid_action = self.scanner.moveLeft(self.speed)
        elif action == 3:
            is_valid_action = self.scanner.moveRight(self.speed)
        elif action == 4:
            is_valid_action = self.scanner.move(np.array([1,1]), self.speed)
        elif action == 5:
            is_valid_action = self.scanner.move(np.array([1,-1]), self.speed)
        elif action == 6:
            is_valid_action = self.scanner.move(np.array([-1,-1]), self.speed)
        elif action == 7:
            is_valid_action = self.scanner.move(np.array([-1,1]), self.speed)
        elif action == 8:
            is_valid_action = self.scanner.lookTo(0)
        elif action == 9:
            is_valid_action = self.scanner.lookTo(45)
        elif action == 10:
            is_valid_action = self.scanner.lookTo(90)
        elif action == 11:
            is_valid_action = self.scanner.lookTo(135)
        elif action == 12:
            is_valid_action = self.scanner.lookTo(180)
        elif action == 13:
            is_valid_action = self.scanner.lookTo(225)
        elif action == 14:
            is_valid_action = self.scanner.lookTo(270)
        elif action == 15:
            is_valid_action = self.scanner.lookTo(315)
        else:
            assert(False)
            
        obs = self._get_obs()
        reward = self._get_reward(self.scanner.scan_map, is_valid_action)
        done = self.ep_terminate

        return obs, reward, done, False, {}

#-----------------------------------------------------------------------------------------------
# Private
#-----------------------------------------------------------------------------------------------
    def _get_obs(self):
        scan_map, _ = self.scanner.scan()
        obs = np.full(self.observation_space.shape, 0.0, dtype=np.float32)
        
        obs[0][np.where(scan_map==UNKOWN)] = 1.0
        obs[1][np.where(scan_map==EMPTY)] = 1.0
        obs[1][np.where(scan_map==IN_VIEW)] = 1.0
        obs[2][np.where(scan_map==SCANNED)] = 1.0
        obs[4][np.where(scan_map==AGENT)] = 1.0
        frontiers = find_frontier(scan_map)
        if len(frontiers) == 0:
            self.clear_scanning = True
        for p in frontiers:
            x, y = p
            obs[3,x,y] = 1.0

        return obs
    
    def _get_reward(self, scan_map:np.ndarray, is_valid:bool):
        if not is_valid:
            self.ep_terminate = True
            return INVALID_ACTION_REWARD
            
        if self.clear_scanning:
            self.ep_terminate = True
            return POS_REWARD

        return NEG_REWARD
            

#-----------------------------------------------------------------------------------------------
# Rendering
#-----------------------------------------------------------------------------------------------
    def render(self):
        buffer = np.full((self.obs_shape[0], self.obs_shape[1], 3), 0, dtype=np.uint8)      # Clear

        # Render
        scan_map = self.scanner.scan_map
        buffer[np.where(scan_map==UNKOWN)] = (0,0,0)
        buffer[np.where(scan_map==EMPTY)] = (64,64,64)
        buffer[np.where(scan_map==SCANNED)] = (255,0,255)
        buffer[np.where(scan_map==IN_VIEW)] = (0,128,128)
        buffer = cv.circle(buffer, ((int(self.scanner.position[1])), int(self.scanner.position[0])), 3, (255,0,0), cv.FILLED)

        obs_buffer = cv.resize(buffer, self.display_shape)
        obs_buffer = np.swapaxes(obs_buffer, 0, 1)
        cv.imshow("Test scanner control", obs_buffer)
        cv.waitKey(16)