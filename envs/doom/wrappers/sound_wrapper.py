import gym
import numpy as np

class DoomSound(gym.Wrapper):
    """Add game variables to the observation space + reward shaping."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space
        self.aud_len = 1260 * 4
        self.unwrapped.skip_frames = 4

        audio_shape = [self.aud_len,2]
        sound_high = [[32767,32767]] * self.aud_len
        sound_low = [[-32767,-32767]] * self.aud_len

        self.observation_space = gym.spaces.Dict({
            'obs': current_obs_space,
            'sound': gym.spaces.Box(
                low=np.array(sound_low, dtype=np.int16), high=np.array(sound_high, dtype=np.int16),
            ),
        })

    

    def reset(self):
        image = self.env.reset()
        # audio = self.unwrapped.game.get_state().audio_buffer
        audio = self.unwrapped.state.audio_buffer

        if audio is None:
            audio = np.zeros(self.observation_space['sound'].shape)

        elif audio.shape[0] != self.aud_len:
            audio = np.zeros(self.observation_space['sound'].shape)


        obs_dict = {
            'obs':image,
            # set to zero and run baselines
            'sound':audio
        }
        return obs_dict

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if not done:
            audio = self.unwrapped.state.audio_buffer
            # audio = self.unwrapped.game.get_state().audio_buffer
        else:
            audio = np.zeros(self.observation_space['sound'].shape)

        obs_dict = {
            'obs':obs,
            'sound':audio
        }
        return obs_dict, rew, done, info
