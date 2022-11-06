#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import configparser
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.airsim_envs.airsim_env import AirSimDroneEnv
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic


class Myconf(configparser.ConfigParser):
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=None)

    def optionxform(self, optionstr: str) -> str:
        return optionstr


def make_train_env(cfg):
    def get_env_fn(rank):
        def init_env():
            if cfg.get('options', 'env') == "airsim":
                env = AirSimDroneEnv(cfg)
            else:
                print("Can not support the " +
                      cfg.get('options', 'env') + "environment.")
                raise NotImplementedError
            env.seed(cfg.getint('algorithm', 'seed') + rank * 1000)
            return env
        return init_env

    if cfg.getint('algorithm', 'n_rollout_threads') == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.getint('algorithm', 'n_rollout_threads'))])


def make_eval_env(cfg):
    def get_env_fn(rank):
        def init_env():
            if cfg.get('options', 'env') == "airsim":
                env = AirSimDroneEnv(cfg)
            else:
                print("Can not support the " +
                      cfg.get('options', 'env') + "environment.")
                raise NotImplementedError
            env.seed(cfg.getint('algorithm', 'seed')*50000 + rank * 10000)
            return env
        return init_env

    if cfg.getint('algorithm', 'n_rollout_threads') == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(cfg.getint('algorithm', 'n_rollout_threads'))])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def _t2n(x):
    return x.detach().cpu().numpy()

# -ddrt --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "zoeyuchao" --user_name "zoeyuchao"
def main(arg):
    cfg = Myconf()
    cfg.read(arg)
    for each in cfg.items("algorithm"):
        cfg.__dict__[each[0]] = each[1]

    if cfg.get('algorithm', 'algorithm_name') == "rmappo":
        assert cfg.getboolean('algorithm', 'use_recurrent_policy') or cfg.getboolean('algorithm', 'use_naive_recurrent_policy'), ("check recurrent policy!")
    elif cfg.get('algorithm', 'algorithm_name')  == "mappo":
        assert cfg.getboolean('algorithm', 'use_recurrent_policy') == False and cfg.getboolean('algorithm', 'use_naive_recurrent_policy') == False, ("check recurrent policy!")
    else:
        raise NotImplementedError


    # cuda

    if cfg.getboolean('algorithm', 'cuda') and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(cfg.getint('algorithm', 'n_training_threads'))
        if cfg.getboolean('algorithm', 'cuda_deterministic'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.getint('algorithm', 'n_training_threads'))

    # seed
    torch.manual_seed(cfg.getint('algorithm', 'seed'))
    torch.cuda.manual_seed_all(cfg.getint('algorithm', 'seed'))
    np.random.seed(cfg.getint('algorithm', 'seed'))


    # env init
    envs = AirSimDroneEnv(cfg)
    eval_envs = make_eval_env(cfg) if cfg.getboolean('algorithm', 'use_eval') else None
    num_agents = cfg.getint('options', 'num_of_drone')

    config = {
        "cfg": cfg,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device
    }

    # load model
    policy_actor_state_dict = torch.load(str(cfg.get("algorithm", 'model_dir')) + '/actor.pt')

    envs.reset()
    actor = R_Actor(config['cfg'], config['envs'].observation_space[0], config['envs'].action_space[0], config['device'])
    actor.load_state_dict(policy_actor_state_dict)
    rnn_states_actor = np.zeros(
        (1, 1, 64),
        dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)

    obs = []
    obs.append(envs._get_obs(envs.agents[0]))
    obs = np.reshape(obs,(1,444))
    while True:

        action, action_log_probs, rnn_states_actor = actor(obs, rnn_states_actor, masks)
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), 1))
        rnn_states_actor = np.reshape(rnn_states_actor, (1,1,64))
        actions = np.array(np.split(_t2n(action), 1))
        actions = np.reshape(actions,(1,1))
        obs, rewards, dones, infos = envs.step(actions)
        obs = np.reshape(obs, (1, 444))
        if envs.agents[0].is_in_desired_pose():
            break


if __name__ == "__main__":
    default_cfg = '../../envs/airsim_envs/cfg/default.cfg'
    if sys.argv[1:]:
        cfg = sys.argv[1:]
    else:
        cfg = default_cfg
    main(default_cfg)
