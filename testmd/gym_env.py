import cv2 as cv
from train.env_wrapper import Scanning_a16

def test_env_wrapper():
    env = Scanning_a16()

    obs, _ = env.reset()
    done = False
    while True:
        key = cv.waitKey()
        if key != -1:
            if key==ord('w'): action = 0
            elif key==ord('s'): action = 1
            elif key==ord('a'): action = 2
            elif key==ord('d'): action = 3

            elif key==ord('u'): action = 8
            elif key==ord('i'): action = 10
            elif key==ord('o'): action = 12
            elif key==ord('p'): action = 14
            obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
