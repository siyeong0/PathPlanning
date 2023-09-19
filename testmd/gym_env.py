import cv2 as cv
from train.env_wrapper import EnvWrapper

def test_env_wrapper():
    env = EnvWrapper()

    obs, _ = env.reset()
    done = False
    while True:
        key = cv.waitKey()
        if key != -1:
            if key==ord('w'): action = 0
            elif key==ord('s'): action = 1
            elif key==ord('a'): action = 2
            elif key==ord('d'): action = 3

            elif key==ord('q'): action = 4
            elif key==ord('e'): action = 5
            obs, reward, done, _, _ = env.step(action)
            print(reward)
        env.render()

        if done:
            obs, _ = env.reset()
