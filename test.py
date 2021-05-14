from random import choice
import numpy as np
import os


from envs.doom.doom_utils import make_doom_env
from algorithms.utils.arguments import default_cfg
from algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env


import cv2
from scipy.io.wavfile import write
import moviepy.editor as mpe
import matplotlib.pyplot as plot


def default_doom_cfg():
    return default_cfg(env='doom_env')

if __name__ == "__main__":

    # env = make_doom_env('doom_sound', cfg=default_doom_cfg(), env_config=None)
    # env = make_doom_env('doom_music_sound_multi', cfg=default_doom_cfg(), env_config=None, custom_resolution = '1280x720')
    env = make_doom_env('doom_duel_bots_sound', cfg=default_doom_cfg(), env_config=None)
    # env = make_doom_env('doom_duel_bots', cfg=default_doom_cfg(), env_config=None)
    # env = make_doom_env('hell_doom_sound_multi', cfg=default_doom_cfg(), env_config=None)
    # env = MultiAgentWrapper(env)
    # env.unwrapped.skip_frames = 1

    
    sleep_time = 1.0 / 35
    sf = env.skip_frames
    sleep_time *= sf


    audios = []
    screens = []

    frames = 0
    episodes = 1


    actions1 = [0,1,2]
    actions2 = [0,1,2]


    for i in range(episodes):
        print("Episode #" + str(i + 1))
        step = 0
        state = env.reset()
        # sampling_freq = env.game.get_sound_sampling_freq()
        # env.game.set_sound_sampling_freq(sampling_freq)
        done = False
        while not env.game.is_episode_finished() or not done:
            if step % 10 == 0:
                ac = [choice(actions1), choice(actions2)]
            next_state, reward, done, info = env.step(ac)
            frames += 1


            if not done and env.unwrapped.state.audio_buffer is not None:
                # audio = state["sound"]
                audio = env.unwrapped.state.audio_buffer
                screen = state["obs"]
                screen = env.unwrapped.state.screen_buffer
                
                list_audio = list(audio)
                # audios.extend(list_audio[:len(list_audio)//4])
                audios.extend(list_audio)
                screens.append(np.swapaxes(np.swapaxes(screen,0,1),1,2)[:,:,::-1])

            state = next_state
            step += 1
            frames += 1

    
    audios = np.array(audios)
    videos = np.array(screens)

    ran = np.random.randint(200)
    os.makedirs("trials/"+str(ran), exist_ok=True)

    plot.specgram(audios[:,0])
    plot.savefig('trials/'+ str(ran) +'/specl.png')
    plot.specgram(audios[:,1])
    plot.savefig('trials/'+ str(ran) +'/specr.png')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('trials/'+ str(ran) +'/video.mp4', fourcc, 35/sf, (videos.shape[2], videos.shape[1]))
    for i in range(len(screens)):
        out.write(screens[i])
    out.release()
    write('trials/'+ str(ran) +'/audio.wav', env.sampling_rate_int, audios)
    # print("total audio time should be :" + str(d))
    my_clip = mpe.VideoFileClip('trials/'+ str(ran) +'/video.mp4')
    audio_background = mpe.AudioFileClip('trials/'+ str(ran) +'/audio.wav')
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("trials/"+ str(ran) +"/movie.mp4")
