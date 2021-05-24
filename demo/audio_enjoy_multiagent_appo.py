import datetime
import math
import os
import sys
import time
from collections import deque
from os.path import join

import numpy as np
import torch

from algorithms.appo.actor_worker import transform_dict_observations
from algorithms.appo.learner import LearnerWorker
from algorithms.appo.model import create_actor_critic
from algorithms.appo.model_utils import get_hidden_size
from algorithms.utils.action_distributions import ContinuousActionDistribution
from algorithms.utils.algo_utils import ExperimentStatus
from algorithms.utils.arguments import parse_args, load_from_checkpoint
from algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from envs.create_env import create_env
from utils.utils import log, AttrDict

import cv2
from scipy.io.wavfile import write
import moviepy.editor as mpe
import matplotlib.pyplot as plot
from PIL import Image

def enjoy(cfg, max_num_episodes=100, max_num_frames=1e9):
    cfg = load_from_checkpoint(cfg)
    cfg.device = 'cpu'
    cfg.train_dir = '/home/khegde/Desktop/Github2/sample-factory/train_dir'

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    cfg.no_render = True

    if cfg.record_to:
        tstamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        cfg.record_to = join(cfg.record_to, f'{cfg.experiment}', tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    # env.seed(0)

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    # baseline model
    actor_critic_b = create_actor_critic(cfg, env.observation_space, env.action_space)

    actor_critic_b.model_to_device(device)

    # checkpoints_b = [
    #     '/home/khegde/Desktop/Github2/sample-factory/train_dir/duel_without_sound_self_play/duel_without_sound_self_play_/00_duel_without_sound_self_play_ppo_1/checkpoint_p0/checkpoint_000490805_2010337280.pth',
    #     '/home/khegde/Desktop/Github2/sample-factory/train_dir/duel_without_sound_self_play/duel_without_sound_self_play_/00_duel_without_sound_self_play_ppo_1/checkpoint_p0/checkpoint_000491042_2011308032.pth',
    #     '/home/khegde/Desktop/Github2/sample-factory/train_dir/duel_without_sound_self_play/duel_without_sound_self_play_/00_duel_without_sound_self_play_ppo_1/checkpoint_p0/checkpoint_000491077_2011451392.pth'
    # ]
    checkpoints_b = checkpoints

    checkpoint_dict_b = LearnerWorker.load_checkpoint(checkpoints_b, device)
    actor_critic_b.load_state_dict(checkpoint_dict_b['model'])    

    episode_rewards = []
    audios = []
    screens = []
    true_rewards = deque([], maxlen=100)
    num_frames = 0

    player1_score = 0
    player2_score = 0
    draws = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()

    with torch.no_grad():
        for _ in range(max_num_episodes):
            done = [False] * len(obs)
            rnn_states = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)
            rnn_states_b = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)

            episode_reward = 0

            while True:
                obs_torch = AttrDict(transform_dict_observations([obs[0]]))
                obs_torch_b = AttrDict(transform_dict_observations([obs[1]]))
                for key, x in obs_torch.items():
                    obs_torch[key] = torch.from_numpy(x).to(device).float()
                for key, x in obs_torch_b.items():
                    obs_torch_b[key] = torch.from_numpy(x).to(device).float()

                # obs_torch['sound'][1] = torch.zeros(obs_torch['sound'][1].shape)
                # obs_torch['obs'][1] = torch.zeros(obs_torch['obs'][1].shape)
                obs_torch_b['sound'] = torch.zeros(obs_torch_b['sound'].shape)
                policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)
                policy_outputs_b = actor_critic_b(obs_torch_b, rnn_states_b, with_action_distribution=True)

                # sample actions from the distribution by default
                actions = policy_outputs.actions
                actions_b = policy_outputs_b.actions

                action_distribution = policy_outputs.action_distribution
                if isinstance(action_distribution, ContinuousActionDistribution):
                    if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                        actions = action_distribution.means

                actions = actions.cpu().numpy()
                actions_b = actions_b.cpu().numpy()

                rnn_states = policy_outputs.rnn_states
                rnn_states_b = policy_outputs_b.rnn_states
                # audio = env.unwrapped.state.audio_buffer
                # if audio is not None:
                #     screen = env.unwrapped.state.screen_buffer
                #     scrn = np.swapaxes(np.swapaxes(screen,0,1),1,2)
                #     screens.append(scrn)                
                #     list_audio = list(audio)
                #     audios.extend(list_audio)

                for _ in range(render_action_repeat):
                    if not cfg.no_render:
                        target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                        current_delay = time.time() - last_render_start
                        time_wait = target_delay - current_delay

                        if time_wait > 0:
                            # log.info('Wait time %.3f', time_wait)
                            time.sleep(time_wait)

                        last_render_start = time.time()
                        env.render()

                    obs, rew, done, infos = env.step(np.array([actions[0], actions_b[0]]))

                    episode_reward += np.mean(rew)
                    num_frames += 1

                    if all(done):
                        true_rewards.append(infos[0].get('true_reward', math.nan))
                        log.info('Episode finished at %d frames', num_frames)
                        if not math.isnan(np.mean(true_rewards)):
                            log.info('true rew %.3f avg true rew %.3f', true_rewards[-1], np.mean(true_rewards))

                        if infos[0].get('FRAGCOUNT') > infos[1].get('FRAGCOUNT'):
                            player1_score += 1
                            log.info('sound agent won!!')
                        elif infos[1].get('FRAGCOUNT') > infos[0].get('FRAGCOUNT'):
                            player2_score += 1
                            log.info('deaf agent won!!')
                        else:
                            draws += 1
                            log.info('Draw!!')

                        # VizDoom multiplayer stuff
                        # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                        #     key = f'PLAYER{player}_FRAGCOUNT'
                        #     if key in infos[0]:
                        #         log.debug('Score for player %d: %r', player, infos[0][key])
                        break

                if all(done) or max_frames_reached(num_frames):
                    break

            if not cfg.no_render:
                env.render()
            time.sleep(0.01)

            episode_rewards.append(episode_reward)
            last_episodes = episode_rewards[-100:]
            avg_reward = sum(last_episodes) / len(last_episodes)
            log.info(
                'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
            )

            if max_frames_reached(num_frames):
                break

    env.close()
    log.info(
            'Final score: sound agent {} - sound agent {} over {} episodes'.format(player1_score, player2_score, max_num_episodes)
        )    # audios = np.array(audios)
    # videos = np.array(screens)

    # ran = np.random.randint(200)
    # os.makedirs("trials/"+str(ran), exist_ok=True)

    # plot.specgram(audios[:,0])
    # plot.savefig('trials/'+ str(ran) +'/specl.png')
    # plot.specgram(audios[:,1])
    # plot.savefig('trials/'+ str(ran) +'/specr.png')

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # # out = cv2.VideoWriter('trials/'+ str(ran) +'/video.mp4', fourcc, 35/env.skip_frames, (128,72))
    # out = cv2.VideoWriter('trials/'+ str(ran) +'/video.mp4', fourcc, 35/(env.skip_frames), (env.screen_w,env.screen_h))
    # for i in range(len(screens)):
    #     out.write(screens[i][:,:,::-1])
    # out.release()
    # write('trials/'+ str(ran) +'/audio.wav', env.sampling_rate_int, audios)
    # # print("total audio time should be :" + str(d))
    # my_clip = mpe.VideoFileClip('trials/'+ str(ran) +'/video.mp4')
    # audio_background = mpe.AudioFileClip('trials/'+ str(ran) +'/audio.wav')
    # final_clip = my_clip.set_audio(audio_background)
    # final_clip.write_videofile("trials/"+ str(ran) +"/movie.mp4")
    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    status, avg_reward = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
