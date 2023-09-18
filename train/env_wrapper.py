import gym
from gym import spaces
import numpy as np
import cv2 as cv

from utils import generate_random_map
from scanner import *
from pathplan import Graph, to_voronoi, extract_contour_graph

REWARD = 1.0
CLEAR_REWARD = 10.0
INVALID_ACTION_REWARD = -10.0

class EnvWrapper(gym.Env):
    def __init__(self, 
                 world_shape=(192,192),
                 obs_shape=(128,128),
                 speed = 4.0,
                 ang_speed = 15,
                 map_gen_options = ["discrete", "straight", "border"],
                 display_shape = (1024,1024)
                 ):
        super(EnvWrapper, self).__init__()
        self.world_shape = world_shape
        #self.obs_shape = obs_shape
        self.obs_shape = world_shape
        self.speed = speed
        self.ang_speed = ang_speed
        self.options = map_gen_options
        self.display_shape = display_shape
    
        self.world = None
        self.scanner = None

        self.num_obs_stack = 1
        self.obs_buffer = None
        self.observation_space = spaces.Box(0, 255, (self.num_obs_stack, self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)
        # self.action_space = spaces.MultiDiscrete([4,3])
        self.action_space = spaces.Discrete(6)

        self.graph = None
        self.target_indices = None
        self.num_target_nodes = None
        self.ep_terminate = None

    def reset(self, seed=None, options=None):
        self.world = generate_random_map(shape=self.world_shape, options=self.options)
        self.scanner = Scanner(self.world, 120, 48)

        self.obs_buffer = [np.zeros(self.obs_shape, dtype=np.uint8) for _ in range(self.num_obs_stack)]
        self.graph = Graph([],[])
        self.target_indices = []
        self.num_target_nodes = 0
        self.ep_terminate = False

        return self._get_obs(), {}
    
    def step(self, action):
        #tr_action, rot_action = action
        is_valid_action = True
        # # Translation
        # if tr_action == 0:
        #     is_valid_action = self.scanner.moveUp(self.speed)
        # elif tr_action == 1:
        #     is_valid_action = self.scanner.moveDown(self.speed)
        # elif tr_action == 2:
        #     is_valid_action = self.scanner.moveLeft(self.speed)
        # elif tr_action == 3:
        #     is_valid_action = self.scanner.moveRight(self.speed)
        # else:
        #     assert(False)
        # # Rotation
        # if rot_action == 0:
        #     pass
        # elif rot_action == 1:
        #     self.scanner.rotate(self.ang_speed)
        # elif rot_action == 2:
        #     self.scanner.rotate(-self.ang_speed)
        # else:
        #     assert(False)
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
        reward = self._get_reward(self.scanner.scan_map, is_valid_action)
        done = self.ep_terminate

        return obs, reward, done, False, {}

#-----------------------------------------------------------------------------------------------
# Private
#-----------------------------------------------------------------------------------------------
    def _get_obs(self):
        pos_x, pos_y = self.scanner.position.astype(int)
        # obs_w, obs_h = self.obs_shape
        # top_left = (pos_x - int(obs_w / 2), pos_y - int(obs_h / 2))
        # bottom_right = (pos_x + int(obs_w / 2), pos_y + int(obs_h / 2))

        # target_rect = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        # offsets = [0 if target_rect[0] >= 0 else -target_rect[0],
        #            0 if target_rect[1] >= 0 else -target_rect[1],
        #            0 if target_rect[2] < self.world_shape[0] else target_rect[2]-(self.world_shape[0]-1),
        #            0 if target_rect[3] < self.world_shape[1] else target_rect[3]-(self.world_shape[1]-1)]
        # obs_rect = [offsets[0], offsets[1], (obs_w-1)-offsets[2], (obs_h-1)-offsets[3]]
        # map_rect = [obs_rect[0]+top_left[0], obs_rect[1]+top_left[1],
        #             obs_rect[2]+top_left[0], obs_rect[3]+top_left[1]]

        scan_map, _ = self.scanner.scan()
        # obs = np.full(self.obs_shape, UNKOWN, dtype=np.uint8)
        # obs[obs_rect[0]:obs_rect[2], obs_rect[1]:obs_rect[3]] = \
        #     scan_map[map_rect[0]:map_rect[2], map_rect[1]:map_rect[3]]
        # obs[pos_x-top_left[0], pos_y-top_left[1]] = AGENT

        scan_map, _ = self.scanner.scan()
        obs = scan_map
        obs[pos_x, pos_y] = AGENT

        obs = obs * int(255 / AGENT)
        for i in range(1,self.num_obs_stack):
            self.obs_buffer[i] = self.obs_buffer[i - 1]
        self.obs_buffer[0] = obs

        self.obs_buffer

        return np.array(self.obs_buffer)
    
    def _get_reward(self, scan_map:np.ndarray, is_valid:bool):
        if not is_valid:
            self.ep_terminate = True
            return INVALID_ACTION_REWARD

        gray_img = np.zeros_like(scan_map, dtype=np.uint8)
        gray_img[np.where(scan_map==SCANNED)] = 255
        contours = extract_contour_graph(gray_img, True)

        valid_map = np.full(scan_map.shape, False, dtype=np.bool_)
        valid_map[np.where(scan_map==EMPTY)] = True
        valid_map[np.where(scan_map==IN_VIEW)] = True

        reward = 0.0

        if len(contours.vertices) > 4:
            self.graph, self.target_indices = to_voronoi(contours, valid_map)

            n_target_nodes = len(self.target_indices)
            if n_target_nodes < self.num_target_nodes:
                reward = REWARD

            self.num_target_nodes = n_target_nodes
            if n_target_nodes == 0:
                self.ep_terminate = True
                reward = CLEAR_REWARD
            
        return reward
            

#-----------------------------------------------------------------------------------------------
# Rendering
#-----------------------------------------------------------------------------------------------
    def render(self):
        world_buffer = np.full((self.world_shape[0], self.world_shape[1], 3), 0, dtype=np.uint8)      # Clear
        obs_buffer = np.full((self.obs_shape[0], self.obs_shape[1], 3), 0, dtype=np.uint8)      # Clear
        
        # Render scan map
        scan_map = self.scanner.scan_map
        scanner_pos = self.scanner.position
        world_buffer[np.where(scan_map == UNKOWN)] = (0,0,0)
        world_buffer[np.where(scan_map == EMPTY)] = (64,64,64)
        world_buffer[np.where(scan_map == SCANNED)] = (255,0,255)
        world_buffer[np.where(scan_map == IN_VIEW)] = (0,128,128)
        world_buffer = self.graph.draw(world_buffer)
        for i in self.target_indices:    # Draw target vertices
            v = self.graph.vertices[i]
            v = (int(v[0]), int(v[1]))
            world_buffer = cv.rectangle(world_buffer, (v[1]-2, v[0]-2), (v[1]+2, v[0]+2), (0,0,255), cv.FILLED)
        world_buffer = cv.circle(world_buffer, (int(scanner_pos[1]), int(scanner_pos[0])), 5, (255,0,0), cv.FILLED)

        # Render observation
        obs = (self._get_obs()[0] /  int(255 / AGENT)).astype(np.uint8)
        obs_buffer[np.where(obs == UNKOWN)] = (0,0,0)
        obs_buffer[np.where(obs == EMPTY)] = (64,64,64)
        obs_buffer[np.where(obs == SCANNED)] = (255,0,255)
        obs_buffer[np.where(obs == IN_VIEW)] = (0,128,128)
        obs_buffer = cv.circle(obs_buffer, (int(self.obs_shape[1]/2), int(self.obs_shape[0]/2)), 3, (255,0,0), cv.FILLED)

        obs_buffer = cv.resize(obs_buffer, self.display_shape)
        minimap_shape = (int(self.display_shape[0] * 0.2), int(self.display_shape[1] * 0.2))
        world_buffer = cv.resize(world_buffer, minimap_shape)
        obs_buffer[self.display_shape[0] - minimap_shape[0]:self.display_shape[0], 0:minimap_shape[1]] = world_buffer
        obs_buffer = cv.rectangle(obs_buffer, (0, self.display_shape[0] - minimap_shape[0]), (minimap_shape[1], self.display_shape[0]), (192,0,0), 1)

        obs_buffer = np.swapaxes(obs_buffer, 0, 1)
        cv.imshow("Test scanner control", cv.resize(obs_buffer, (512,512)))
        cv.waitKey(16)