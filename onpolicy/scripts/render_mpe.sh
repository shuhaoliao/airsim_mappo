#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=1 python render/render_mpe.py --save_gifs --share_policy --env_name MPE --algorithm_name rmappo --experiment_name check --scenario_name simple_speaker_listener --num_agents 2 --num_landmarks 3 --seed 1 --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 --model_dir "D:\Airsim_attack"
done
