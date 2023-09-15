import cv2 as cv
from train.env_wrapper import EnvWrapper

def test_env_wrapper():
    env = EnvWrapper()

    obs, _ = env.reset()
    done = True
    while True:
        key = cv.waitKey()
        if key != -1:
            tr_action, rot_action = 0, 0
            if key==ord('w'): tr_action = 1
            elif key==ord('s'): tr_action = 2
            elif key==ord('a'): tr_action = 3
            elif key==ord('d'): tr_action = 4

            elif key==ord('q'): rot_action = 1
            elif key==ord('e'): rot_action = 2
            obs, reward, done, _, _ = env.step((tr_action, rot_action))
            print(reward)
        env.render()

        if done:
            obs, _ = env.reset()
