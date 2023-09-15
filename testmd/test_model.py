import cv2 as cv
from train.env_wrapper import EnvWrapper
from stable_baselines3 import A2C

def test_model(name:str):
    model = A2C.load(f"./checkpoints/{name}/{name}")

    env = EnvWrapper()

    obs, _ = env.reset()
    done = True
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            obs, _ = env.reset()
