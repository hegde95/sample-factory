#!/usr/bin/env bash
python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 500000000 --num_workers 24 --num_envs_per_worker 20 --env doom_sound_multi --experiment doom_sound_multi_1 --encoder_custom vizdoomSound

python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 500000000 --num_workers 24 --num_envs_per_worker 20 --env doom_sound_multi --experiment doom_sound_multi_2 --encoder_custom vizdoomSound

python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 500000000 --num_workers 24 --num_envs_per_worker 20 --env doom_sound_multi --experiment doom_sound_multi_3 --encoder_custom vizdoomSound
