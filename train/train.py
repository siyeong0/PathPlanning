import gym
from train.env_wrapper import Scanning_a6, Scanning_a16

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import torch
from train.netwrok import BasicNet, Conv7K3
from train.lr_scheduler import linear_schedule
import matplotlib.pyplot as plt
from utils.log_viewer import plot_csv

def train_dqn():
    # ------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------
    name = "dqn_bs_01"
    num_timesteps = 1000000
    net_arch = [128]
    basic_kwargs = dict(features_extractor_class=BasicNet, activation_fn=torch.nn.ReLU, net_arch=net_arch)
    log_path = f"./checkpoints/{name}/log/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        save_freq=int(50000),
        save_path=f"./checkpoints/{name}/cp/",
        name_prefix=name,
        save_replay_buffer=False,
        save_vecnormalize=False)
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    env = Scanning_a6()

    model = DQN("CnnPolicy", env, policy_kwargs=basic_kwargs, 
                    gamma=0.99, 
                    buffer_size=30000, 
                    batch_size=512,
                    #tau=0.05,
                    learning_starts=2000,
                    exploration_initial_eps=0.5,
                    exploration_final_eps=0.05,
                    exploration_fraction=0.01,
                    learning_rate=linear_schedule(9e-3, 1e-4),
                    device= "cuda" if torch.cuda.is_available() else "cpu")

    model.set_logger(new_logger)
    model = model.learn(total_timesteps=num_timesteps, 
                        callback=checkpoint_callback)

    # ------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------
    model.save(f"./checkpoints/{name}/{name}")

    # ------------------------------------------------------------
    # Plot result
    # ------------------------------------------------------------
    plt.xlabel('Num Step')
    plt.ylabel('Reward Avg')
    data = plot_csv(f"./checkpoints/{name}/log/progress.csv", label="rollout/ep_rew_mean", verbose=False, ret_data=True)
    plt.plot(data[0], data[1], label=name)
    plt.legend()

    plt.savefig(f"./checkpoints/{name}/{name}_rew.png")
    plt.show()

def train_a2c():
    # ------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------
    name = "a2c_c7_01"
    pt_name = ""
    num_envs = 16
    num_timesteps = 1000000
    net_arch = dict(pi=[128, 64], vf=[128, 64])
    basic_kwargs = dict(features_extractor_class=Conv7K3, activation_fn=torch.nn.ReLU, net_arch=net_arch)
    log_path = f"./checkpoints/{name}/log/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        save_freq=int(50000 / num_envs),
        save_path=f"./checkpoints/{name}/cp/",
        name_prefix=name,
        save_replay_buffer=False,
        save_vecnormalize=False)
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    env = make_vec_env(Scanning_a6, n_envs=num_envs, seed=1)

    model = A2C("CnnPolicy", env, policy_kwargs=basic_kwargs, 
                    gamma=0.99, 
                    n_steps=64,
                    ent_coef=0.0,
                    gae_lambda=0.95,
                    learning_rate=linear_schedule(7e-4, 7e-4),
                    device= "cuda" if torch.cuda.is_available() else "cpu")
    # Load pretrained model
    if pt_name != "" and pt_name != None:
        pt_model = A2C.load(f"./checkpoints/{pt_name}/{pt_name}", env=env, 
                            device="cuda" if torch.cuda.is_available() else "cpu")
        model.policy.features_extractor = pt_model.policy.features_extractor
        model.policy.mlp_extractor = pt_model.policy.mlp_extractor
        model.policy.action_net = pt_model.policy.action_net
        model.policy.value_net = pt_model.policy.value_net
    model.set_logger(new_logger)

    model.policy.features_extractor.training = False
    model.policy.mlp_extractor.training = False
    model = model.learn(total_timesteps=num_timesteps, 
                        callback=checkpoint_callback)

    # ------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------
    model.save(f"./checkpoints/{name}/{name}")

    # ------------------------------------------------------------
    # Plot result
    # ------------------------------------------------------------
    plt.xlabel('Num Step')
    plt.ylabel('Reward Avg')
    data = plot_csv(f"./checkpoints/{name}/log/progress.csv", label="rollout/ep_rew_mean", verbose=False, ret_data=True)
    plt.plot(data[0], data[1], label=name)
    plt.legend()

    plt.savefig(f"./checkpoints/{name}/{name}_rew.png")
    plt.show()

def train_ppo():
    # ------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------
    name = "ppo_bs_02_i2"
    pt_name = "ppo_bs_02_i0"
    num_envs = 16
    num_timesteps = 1500000
    net_arch = dict(pi=[128], vf=[128])
    basic_kwargs = dict(features_extractor_class=BasicNet, activation_fn=torch.nn.ReLU, net_arch=net_arch)
    log_path = f"./checkpoints/{name}/log/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        save_freq=int(50000 / num_envs),
        save_path=f"./checkpoints/{name}/cp/",
        name_prefix=name,
        save_replay_buffer=False,
        save_vecnormalize=False)
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    env = make_vec_env(Scanning_a16, n_envs=num_envs, seed=1)

    model = PPO("CnnPolicy", env, policy_kwargs=basic_kwargs, 
                    learning_rate=linear_schedule(4e-3, 4e-5),
                    gamma=0.99, 
                    n_epochs=10,
                    batch_size=2048,
                    n_steps=2048,
                    vf_coef=0.5,
                    ent_coef=0.0,
                    gae_lambda=0.95,
                    clip_range=0.3,
                    device= "cuda" if torch.cuda.is_available() else "cpu")
    # Load pretrained model
    if pt_name != "" and pt_name != None:
        pt_model = PPO.load(f"./checkpoints/{pt_name}/{pt_name}", env=env, 
                            device="cuda" if torch.cuda.is_available() else "cpu")
        model.policy.features_extractor = pt_model.policy.features_extractor
        model.policy.mlp_extractor = pt_model.policy.mlp_extractor
        model.policy.action_net = pt_model.policy.action_net
        model.policy.value_net = pt_model.policy.value_net
    model.set_logger(new_logger)

    model.policy.features_extractor.training = False
    model.policy.mlp_extractor.training = False
    model = model.learn(total_timesteps=num_timesteps, 
                        callback=checkpoint_callback)

    # ------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------
    model.save(f"./checkpoints/{name}/{name}")

    # ------------------------------------------------------------
    # Plot result
    # ------------------------------------------------------------
    plt.xlabel('Num Step')
    plt.ylabel('Reward Avg')
    data = plot_csv(f"./checkpoints/{name}/log/progress.csv", label="rollout/ep_rew_mean", verbose=False, ret_data=True)
    plt.plot(data[0], data[1], label=name)
    plt.legend()

    plt.savefig(f"./checkpoints/{name}/{name}_rew.png")
    plt.show()