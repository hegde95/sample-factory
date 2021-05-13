from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    # ('seed', [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
    ('seed', [0, 1111, 2222]),
    ('env', ['doom_music_sound_multi']),
    ('encoder_custom', ['vizdoom', 'vizdoomSoundSamples', 'vizdoomSoundLogMel', 'vizdoomSoundFFT']),
])

_experiments = [
    Experiment(
        'multi_sound_basic',
        'python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 100000000 --num_workers 18 --num_envs_per_worker 20 --experiment multisound_hell_vizdoomSoundFFT_3',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_multi_sound_basic', experiments=_experiments)
