import cv2 as cv
from train.env_wrapper import Scanning_a16
from stable_baselines3 import DQN, A2C, PPO

def test_model(name:str):
    model = PPO.load(f"./checkpoints/{name}/{name}")

    env = Scanning_a16()

    obs, _ = env.reset()
    done = False
    reward_sum = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        reward_sum += reward

        print(reward_sum)

        if done:
            obs, _ = env.reset()
            reward_sum = 0.0
