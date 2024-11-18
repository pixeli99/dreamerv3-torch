from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.utils import generate_gif
from IPython.display import Image, clear_output
import cv2

from metadrive import TopDownMetaDrive
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
import matplotlib.pyplot as plt

# map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM, 
#             BaseMap.GENERATE_CONFIG: 3,  # 3 block
#             BaseMap.LANE_WIDTH: 3.5,
#             BaseMap.LANE_NUM: 2}
# map_config["config"]=3

# env=MetaDriveEnv(dict(map_config=map_config,
#                       agent_policy=ExpertPolicy,
#                       log_level=50,
#                       num_scenarios=1000,
#                       start_seed=0,
#                       traffic_density=0.2))
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
nuscenes_data=AssetLoader.file_path("/18940970966/nuplan_mini", "meta_drive", unix_style=False) 
env = TopDownScenarioEnvV2(
    {
        "reactive_traffic": False,
        "use_render": False,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": nuscenes_data,
        "num_scenarios": 1800,
    }
)

names = [
        "road_network", "past_pos", "traffic t", "traffic t-1", "traffic t-2", "traffic t-3",
        "traffic t-4"
    ]

try:
    # run several episodes
    env.reset(seed=12)

    policy = env.engine.get_policy(env.current_track_agent.name)
    for step in range(500):
        # simulation
        action = policy.get_action_info()
        print(action)
        obs,_,_,_,info = env.step([0, 3])
        env.render(mode="topdown", 
                   window=False,
                   screen_record=True,
                   film_size=(1600, 1600),
                   screen_size=(512, 512),
                   scaling=None,
                   draw_contour=True,
                   num_stack=0,
                   target_agent_heading_up=False
                  )
        fig, axes = plt.subplots(1, obs.shape[-1], figsize=(15, 3))
        for o_i in range(obs.shape[-1]):
            axes[o_i].imshow(obs[..., o_i], cmap="gray", vmin=0, vmax=1)
            axes[o_i].set_title(names[o_i])

        fig.suptitle("Multi-channel Top-down Observation")
        plt.savefig("test.png")
        if info["arrive_dest"]:
            break
    env.top_down_renderer.generate_gif()
finally:
    env.close()