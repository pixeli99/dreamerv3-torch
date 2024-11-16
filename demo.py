import argparse
import os
import pathlib
import sys
import torch
import numpy as np
import ruamel.yaml as yaml
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
from torch.utils.tensorboard import SummaryWriter
import tools
import models
from utils import ReplayBuffer

import gym
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

sys.path.append(str(pathlib.Path(__file__).parent))
os.environ['SDL_VIDEODRIVER'] = 'dummy'

to_np = lambda x: x.detach().cpu().numpy()

def make_env(config):
    # Configure MetaDrive environment
    nuscenes_data = AssetLoader.file_path("/18940970966/nuplan_mini", "meta_drive", unix_style=False)
    env = ScenarioEnv(
        {
            "reactive_traffic": False,
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,  # Only for rendering during inference
            "data_directory": nuscenes_data,
            "num_scenarios": 1800,
            "sequential_seed": True
        }
    )
    return env

def main(config):
    # Create environment
    env = make_env(config)

    # Load the world model
    shape = (128, 128, 3)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    world_model = models.WorldModel(
        obs_space=gym.spaces.Dict({"image": space}),
        act_space=None,
        step=0,
        config=config
    ).to(config.device)

    world_model.load_state_dict(torch.load(config.load_from_checkpoint))
    print(f"Model loaded from {config.load_from_checkpoint}")

    # Initialize replay buffer
    buffer = ReplayBuffer(config.buffer_size, config.batch_length)
    # Sampling data for video prediction
    while len(buffer) < 300:
        obs, _ = env.reset()
        policy = env.engine.get_policy(env.current_track_agent.name)
        terminated = False
        episode_steps = 0
        while not terminated:
            obs_rgb = env.render(
                mode="topdown",
                window=False,
                screen_record=True,
                film_size=(1600, 1600),
                screen_size=(128, 128),
                scaling=4,
                draw_contour=False,
                num_stack=0,
                target_agent_heading_up=True
            )
            
            policy_action = policy.get_action_info()
            velocity = policy_action["velocity"]
            angular_velocity = policy_action["angular_velocity"]
            action = np.array(velocity.tolist() + [angular_velocity])
            
            is_first = episode_steps == 0
            obs, reward, terminated, truncated, info = env.step([0, 3])
            is_terminal = terminated or truncated
            buffer.add(obs_rgb, action, reward, is_terminal, is_first)
            episode_steps += 1

    batch = buffer.sample(config.batch_size)
    obs_batch, action_batch, reward_batch, is_first_batch, is_terminal_batch = batch

    data = {
        'image': obs_batch,
        'action': action_batch,
        'reward': reward_batch,
        'is_first': is_first_batch,
        'is_terminal': is_terminal_batch,
    }

    video_pred = world_model.video_pred(data)
    log_dir = pathlib.Path('infer_demo_logs').expanduser()
    logger = tools.Logger(log_dir, step=0)
    logger.video("infer_openl", to_np(video_pred))
    logger.write()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="wm_mini.yaml")
    args, remaining = parser.parse_known_args()

    # Load configuration file
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / args.config_file).read_text())

    # Parse configuration
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)

    main(config)