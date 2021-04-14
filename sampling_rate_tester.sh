#!/usr/bin/env bash
python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 100000000 --num_workers 24 --num_envs_per_worker 20 --env doom_sound --experiment sr_11025_1 --encoder_custom vizdoomSound

python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 100000000 --num_workers 24 --num_envs_per_worker 20 --env doom_sound --experiment sr_11025_2 --encoder_custom vizdoomSound

python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 100000000 --num_workers 24 --num_envs_per_worker 20 --env doom_sound --experiment sr_11025_3 --encoder_custom vizdoomSound
