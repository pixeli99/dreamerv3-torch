import torch
import numpy as np
from collections import deque

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel

import os
import pathlib
import pickle
import random

class ReplayBuffer:
    def __init__(self, buffer_size, sequence_length, save_dir="replay_buffer"):
        """
        A replay buffer for storing and sampling fixed-length sequences, with optional local saving.

        Args:
            buffer_size (int): Maximum size of the buffer.
            sequence_length (int): Length of the sequences to sample.
            save_dir (str): Directory for saving completed trajectories.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.current_episode = []  # Temporarily store the current episode

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, done, is_first):
        """
        Add an observation, action, reward, done flag, and is_first flag to the current episode buffer.
        If `done=True`, save the complete episode to disk.

        Args:
            obs: The observation at the current timestep.
            action: The action taken at the current timestep.
            reward: The reward received at the current timestep.
            done (bool): Whether the episode is done.
            is_first (bool): Whether this is the first step of an episode.
        """
        step = (obs, action, reward, done, is_first)
        self.current_episode.append(step)

        if done:  # If the episode is complete
            self._save_episode(self.current_episode)
            self.buffer.append(self.current_episode)  # Store in memory buffer
            self.current_episode = []  # Reset the current episode

    def _save_episode(self, episode):
        """
        Save a complete episode to disk as a pickle file.

        Args:
            episode (list): A list of steps in the episode.
        """
        episode_id = len(list(self.save_dir.glob("*.pkl")))  # Use file count as ID
        file_path = self.save_dir / f"episode_{episode_id:05d}.pkl"
        with file_path.open("wb") as f:
            pickle.dump(episode, f)

    def load_episodes(self, limit=None):
        """
        Load saved episodes from disk into the memory buffer.

        Args:
            limit (int): The maximum number of episodes to load. If None, load all.

        Returns:
            List of loaded episodes.
        """
        self.buffer.clear()
        # episodes = []
        # for i, file_path in enumerate(sorted(self.save_dir.glob("*.pkl"))):
        #     if limit and i >= limit:
        #         break
        #     with file_path.open("rb") as f:
        #         episodes.append(pickle.load(f))
        # self.buffer.extend(episodes)  # Add loaded episodes to memory buffer
        # return episodes
        # Collect all file paths
        file_paths = list(self.save_dir.glob("*.pkl"))

        # Shuffle file paths
        random.shuffle(file_paths)

        # episodes = []
        for i, file_path in enumerate(file_paths):
            if limit and i >= limit:
                break
            with file_path.open("rb") as f:
                self.buffer.extend(pickle.load(f))
        return

    def sample(self, batch_size, seed=0):
        """
        Sample a batch of sequences from the buffer, ensuring sequences start at `is_first=True`.

        Args:
            batch_size (int): Number of sequences to sample.
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Arrays containing observations, actions, rewards, is_first flags, and terminal flags.
        """
        if len(self.buffer) < self.sequence_length:
            raise ValueError("Not enough data in buffer to sample a full sequence.")

        np_random = np.random.RandomState(seed)
        ret_obs, ret_actions, ret_rewards, ret_is_first, ret_done = [], [], [], [], []

        while len(ret_obs) < batch_size:
            # Randomly pick a valid starting index
            start_idx = np_random.randint(0, len(self.buffer) - self.sequence_length + 1)
            episode = {
                "obs": [],
                "action": [],
                "reward": [],
                "is_first": [],
                "done": [],
            }

            # Collect sequences from the buffer
            for i in range(self.sequence_length):
                obs, action, reward, done, is_first = self.buffer[start_idx + i]
                episode["obs"].append(obs.numpy() if isinstance(obs, torch.Tensor) else obs)
                episode["action"].append(action)
                episode["reward"].append(reward)
                episode["is_first"].append(is_first)
                episode["done"].append(done)

            # Manually set the first element's `is_first` to True
            episode["is_first"][0] = True

            ret_obs.append(episode["obs"])
            ret_actions.append(episode["action"])
            ret_rewards.append(episode["reward"])
            ret_is_first.append(episode["is_first"])
            ret_done.append(episode["done"])

        # Convert to numpy arrays
        return (
            np.array(ret_obs),
            np.array(ret_actions),
            np.array(ret_rewards),
            np.array(ret_is_first),
            np.array(ret_done),
        )


class TopDownScenarioEnvV2(ScenarioEnv):
    @classmethod
    def default_config(cls):
        config = ScenarioEnv.default_config()
        config["vehicle_config"]["lidar"] = {"num_lasers": 0, "distance": 0}  # Remove lidar
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 4,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 128,
                "distance": 30
            }
        )
        return config

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["use_render"],
            self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )