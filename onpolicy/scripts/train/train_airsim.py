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
"""Train script for MPEs."""


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

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / cfg.get('options', 'env') / \
              cfg.get('options', 'env_name') / cfg.get('algorithm', 'algorithm_name')

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if cfg.get('algorithm', 'use_wandb'):
        run = wandb.init(project=cfg.get('options', 'env'),
                         entity=cfg.get('options', 'user_name'),
                         notes=socket.gethostname(),
                         name=str(cfg.get('algorithm', 'algorithm_name') + "_" +
                                  "seed_" + str(cfg.getint('algorithm', 'seed'))),
                         group=cfg.get('options', 'env_name') ,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(cfg.get('algorithm', 'algorithm_name')) + "-" + \
                              str(cfg.get('options', 'env')) + "-" + str(cfg.get('options', 'env_name')) + "@" + str(
        cfg.get('options', 'user_name')))

    # seed
    torch.manual_seed(cfg.getint('algorithm', 'seed'))
    torch.cuda.manual_seed_all(cfg.getint('algorithm', 'seed'))
    np.random.seed(cfg.getint('algorithm', 'seed'))

    # env init
    envs = make_train_env(cfg)
    eval_envs = make_eval_env(cfg) if cfg.getboolean('algorithm', 'use_eval') else None
    num_agents = cfg.getint('options', 'num_of_drone')

    config = {
        "cfg": cfg,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if cfg.getboolean('algorithm', 'share_policy'):
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if cfg.getboolean('algorithm', 'use_eval') and eval_envs is not envs:
        eval_envs.close()

    if cfg.getboolean('algorithm', 'use_wandb'):
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    default_cfg = '../../envs/airsim_envs/cfg/default.cfg'
    if sys.argv[1:]:
        cfg = sys.argv[1:]
    else:
        cfg = default_cfg
    main(default_cfg)
