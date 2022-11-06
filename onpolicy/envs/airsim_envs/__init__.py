from .airsim_env import AirSimDroneEnv
from gym.envs.registration import register


# Register AirSim environment as a gym environment
register(
    id="airsim-env-v0", entry_point="scripts:AirSimDroneEnv",
)