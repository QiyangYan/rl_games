import time
import math
import torch
import numpy as np
import os


class Data_Sampler(object):
    def __init__(self, device, reward_tune='no', split_ratio=0.8, shuffle=True):

        data = get_data()
        
        self.state = torch.from_numpy(data['observations']).float()
        self.goal = torch.from_numpy(data['desired_goals']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()
        
        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.reward = reward

        # **Shuffle and Split Data**
        indices = np.arange(self.size)
        if shuffle:
            np.random.shuffle(indices)

        split_idx = int(self.size * split_ratio)
        self.train_idx = indices[:split_idx]
        self.test_idx = indices[split_idx:]

    def sample_idxs(self, batch_size, train=True):
        """Samples a batch from training or testing data."""
        idx = np.random.choice(self.train_idx if train else self.test_idx, size=batch_size, replace=True)
        return (
            self.state[idx].to(self.device),
            self.action[idx].to(self.device),
        )

    def get_train_sampler(self):
        """Returns a sampler for training data."""
        return Data_Sampler_Subset(self, self.train_idx)

    def get_test_sampler(self):
        """Returns a sampler for testing data."""
        return Data_Sampler_Subset(self, self.test_idx)


class Data_Sampler_Subset:
    """Helper class for train and test samplers."""
    def __init__(self, parent, indices):
        self.parent = parent
        self.indices = indices

    def sample(self, batch_size):
        idx = np.random.choice(self.indices, size=batch_size, replace=True)
        return (
            self.parent.state[idx].to(self.parent.device),
            self.parent.action[idx].to(self.parent.device),
        )


def get_data():
    """
    Loads and merges all trajectories from the 'demo_slow' directory.
    """
    data_dir = "/home/zhiyuan/Downloads/demo_slow" # safe RL teacher trajectory from Zeyuan's Reward Shaping
    # data_dir = "/home/zhiyuan/Downloads/demo_heuristic_25"

    merged_data = {
        "observations": [],
        "desired_goals": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
    }

    for file_name in sorted(os.listdir(data_dir)):  
        if file_name.endswith(".npy"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Loading: {file_path}")

            data_origin = np.load(file_path, allow_pickle=True).item()
            processed_data = process_single_trajectory(data_origin)

            for key in merged_data.keys():
                merged_data[key].append(processed_data[key])

    for key in merged_data.keys():
        if len(merged_data[key]) > 0:
            merged_data[key] = np.concatenate(merged_data[key], axis=0)
        else:
            merged_data[key] = np.array([])  # Handle empty case
    
    return merged_data


def process_single_trajectory(data_origin):
    """
    Processes a single trajectory from a loaded .npy file.
    """
    actions = data_origin["raw_actions"][1:]
    obs = data_origin['full_states'][:-1]

    data = {
        "observations": obs,
        "desired_goals": np.zeros_like(obs),
        "actions": actions,
        "next_observations": np.zeros_like(obs),
        "rewards": np.zeros((obs.shape[0], 1)),
        "terminals": np.zeros((obs.shape[0], 1)),
    }
    return data


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
    reward /= (rt_max - rt_min)
    reward *= 1000.
    return reward
