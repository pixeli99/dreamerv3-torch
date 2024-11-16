import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, sequence_length):
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.buffer)
        
    def add(self, obs, action, reward, done, is_first):
        """Add an observation, action, and done flag to the buffer."""
        self.buffer.append((obs, action, reward, done, is_first))
        
    def sample(self, batch_size):
        if len(self.buffer) < self.sequence_length:
            raise ValueError("Not enough data in buffer to sample a full sequence.")
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        is_first_batch = []
        is_terminal_batch = []
        valid_indices = []

        # Identify valid start indices for sequences that do not cross episode boundaries
        for idx in range(len(self.buffer) - self.sequence_length):
            is_valid = True
            for i in range(self.sequence_length - 1):
                _, _, _, done_flag, _ = self.buffer[idx + i]
                if done_flag:  # Check if the sequence crosses an episode boundary
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(idx)
        
        if len(valid_indices) < batch_size:
            raise ValueError("Not enough valid sequences to sample the requested batch size.")
        
        # Randomly select start indices from valid ones
        indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        for idx in indices:
            obs_seq = []
            action_seq = []
            reward_seq = []
            is_first_seq = []
            is_terminal_seq = []
            for i in range(self.sequence_length):
                obs, action, reward, done_flag, is_first = self.buffer[idx + i]
                obs_seq.append(obs.numpy() if isinstance(obs, torch.Tensor) else obs)
                action_seq.append(action)
                reward_seq.append(reward)
                is_first_seq.append(is_first)
                is_terminal_seq.append(done_flag)
            
            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            is_first_batch.append(is_first_seq)
            is_terminal_batch.append(is_terminal_seq)
        
        # Convert to numpy arrays for further processing
        return (
            np.array(obs_batch),
            np.array(action_batch),
            np.array(reward_batch),
            np.array(is_first_batch),
            np.array(is_terminal_batch)
        )
