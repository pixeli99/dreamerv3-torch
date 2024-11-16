import argparse
import os
import pathlib
import sys
import torch
import gym
import numpy as np
import ruamel.yaml as yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
from utils import ReplayBuffer
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.engine.asset_loader import AssetLoader

os.environ['SDL_VIDEODRIVER'] = 'dummy'

to_np = lambda x: x.detach().cpu().numpy()

def make_env(config):
    # 配置 MetaDrive 环境
    nuscenes_data = AssetLoader.file_path("/18940970966/nuplan_mini", "meta_drive", unix_style=False)
    env = ScenarioEnv(
        {
            "reactive_traffic": False,
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            "data_directory": nuscenes_data,
            "num_scenarios": 1800,
            "sequential_seed": True
        }
    )
    return env

def main(config):
    # 设置随机种子
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    # 创建日志目录
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, step=0)

    # 创建环境
    print("Create environment.")
    env = make_env(config)

    # 定义世界模型
    shape = (128, 128, 3)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    world_model = models.WorldModel(
        obs_space=gym.spaces.Dict({"image": space}),
        act_space=None,
        step=0,
        config=config
    ).to(config.device)
    
    if config.load_from_checkpoint:
        world_model.load_state_dict(torch.load(config.load_from_checkpoint))
        print(f"Model loaded from {config.load_from_checkpoint}")
    
    if config.compile:
        world_model = torch.compile(world_model)

    # 定义重放缓冲区
    buffer = ReplayBuffer(config.buffer_size, config.batch_length)

    # 训练循环
    global_step = 0
    metrics = {}
    
    for episode in range(config.train_episodes):
        obs, _ = env.reset()
        terminated = False
        episode_steps = 0
        # 获取策略（这里使用环境自带的策略）
        policy = env.engine.get_policy(env.current_track_agent.name)

        obs, reward, terminated, truncated, info = env.step([0, 3])
        
        while not terminated:
            # 渲染环境，获取图像观测
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
            # 获取动作
            policy_action = policy.get_action_info()
            velocity = policy_action["velocity"]
            angular_velocity = policy_action["angular_velocity"]
            action = np.array(velocity.tolist() + [angular_velocity])

            is_first = episode_steps == 0
            obs, reward, terminated, truncated, info = env.step([0, 3])
            
            is_terminal = terminated or truncated
            buffer.add(obs_rgb, action, reward, is_terminal, is_first)
            episode_steps += 1
            # 将数据添加到缓冲区

        # 检查缓冲区大小
        if len(buffer) < 500:
            continue

        # 从缓冲区采样数据进行训练
        for _ in range(config.train_iters):
            global_step += 1
            batch = buffer.sample(config.batch_size)
            obs_batch, action_batch, reward_batch, is_first_batch, is_terminal_batch = batch

            # 准备数据
            data = {
                'image': obs_batch,
                'action': action_batch,
                'reward': reward_batch,
                'is_first': is_first_batch,
                'is_terminal': is_terminal_batch,
            }

            post, context, mets = world_model._train(data)
            
            for name, value in mets.items():
                if name not in metrics:
                    metrics[name] = [value]
                else:
                    metrics[name].append(value)

            if global_step % config.log_every == 0:
                for name, values in metrics.items():
                    logger.scalar(name, float(np.mean(values)))
                    metrics[name] = []  # 清空指标列表
                logger.step = global_step
                logger.write(fps=True)
            
            # 保存模型
            if global_step % config.save_every == 0:
                save_path = os.path.join(logdir, f"model_step_{global_step}.pth")
                torch.save(world_model.state_dict(), save_path)
                print(f"Model saved at {save_path}")

            if config.video_pred_log and global_step % config.video_log_every == 0:
                # 从缓冲区采样数据
                batch = buffer.sample(config.batch_size)
                obs_batch, action_batch, reward_batch, is_first_batch, is_terminal_batch = batch

                # 准备数据
                data = {
                    'image': obs_batch,
                    'action': action_batch,
                    'reward': reward_batch,
                    'is_first': is_first_batch,
                    'is_terminal': is_terminal_batch,
                }

                # 生成视频预测
                video_pred = world_model.video_pred(data)
                logger.video("train_openl", to_np(video_pred))
                logger.write()
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="wm.yaml")
    args, remaining = parser.parse_known_args()

    # 加载配置文件
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / args.config_file).read_text()
    )

    # 解析配置
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