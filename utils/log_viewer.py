import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_csv(path, label='rollout/ep_rew_mean', domain='time/total_timesteps', verbose=True, ret_data = False):
    df = pd.read_csv(path)

    table = {}
    for i, c in enumerate(df):
        table[c] = i

    time_step_buf = []
    target_buf = []
    for val in df.values:
        target_buf.append(val[table[label]])
        time_step_buf.append(val[table[domain]])

    if ret_data == True:
        return (np.array(time_step_buf), np.array(target_buf))

    plt.plot(np.array(time_step_buf), np.array(target_buf))
    name = os.path.basename(path[:-(len(os.path.basename(path))+1)])
    plt.title(name)
    plt.xlabel("Time Step")
    plt.ylabel(os.path.basename(label))
    plt.savefig(f"{os.path.dirname(path)}/{os.path.basename(label)}_fig.png")
    if(verbose):
        plt.show()