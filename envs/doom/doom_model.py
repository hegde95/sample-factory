import torch
from torch import nn
import torchaudio
import torch.nn.functional as F

from algorithms.appo.model_utils import get_obs_shape, nonlinearity, create_standard_encoder, EncoderBase, \
    register_custom_encoder
from algorithms.utils.pytorch_utils import calc_num_elements
from utils.utils import log


SAMPLE_RATE = 44100


class VizdoomEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.measurements_head = None
        if 'measurements' in obs_shape:
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_shape.measurements[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_shape.measurements)
            self.encoder_out_size += measurements_out_size

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict)

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict['measurements'].float())
            x = torch.cat((x, measurements), dim=1)

        return x



class VizdoomSoundEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)
        # obs_shape.sound is 5040 x 2

        self.sound_head = None
        if 'sound' in obs_shape:
            self.sound_head = nn.Sequential(
                LogMelAudioEncoder(),
                nn.Flatten(),
                nn.ReLU(),

                #AudioEncoder(),
                #nn.Flatten(),
                #nn.Linear(4800, 64),
                #nn.ReLU(),

                #nn.Flatten(),
                #nn.Linear(obs_shape.sound[0]*obs_shape.sound[1], 128),
                #nn.ReLU(),
                #nn.Linear(128, 32),
                #nn.ReLU(),
            )
            sound_out_size = calc_num_elements(self.sound_head, obs_shape.sound)
            self.encoder_out_size += sound_out_size

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict)

        
        if self.sound_head is not None:

            # Normalize to [-1, 1] (default for audio)
            obs_dict['sound'].mul_(1.0 / 32767)

            sound = self.sound_head(obs_dict['sound'].float())
            
            x = torch.cat((x, sound), dim=1)

        return x

class LogMelAudioEncoder(nn.Module):
    def __init__(self):
        super(LogMelAudioEncoder, self).__init__()

        # Anssi: These are just parameters I took from
        #        state-of-the-art-ish speaker recognition system
        #        https://www.isca-speech.org/archive/Odyssey_2020/pdfs/65.pdf
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=80,
            n_fft=int(SAMPLE_RATE * 0.025),
            win_length=int(SAMPLE_RATE * 0.025),
            hop_length=int(SAMPLE_RATE * 0.01),
            f_min=20,
            f_max=7600,
        )

        #Encoder
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = x[:,:,0]
        x2 = x[:,:,1]

        # Left channel encoder
        x1 = torch.log(self.mel_spectrogram(x1) + 1e-5)
        x1 = torch.reshape(x1, (x1.shape[0], 1, x1.shape[1], x1.shape[2]))
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool(x1)
        # x1 = F.relu(self.conv3(x1))


        # Right channel encoder
        x2 = torch.log(self.mel_spectrogram(x2) + 1e-5)
        x2 = torch.reshape(x2, (x2.shape[0], 1, x2.shape[1], x2.shape[2]))
        x2 = F.relu(self.conv1(x2))
        x2 = self.pool(x2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.pool(x2)
        # x2 = F.relu(self.conv3(x2))


        x = torch.cat((x1,x2), dim=1)
        return x

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.spec = torchaudio.transforms.Spectrogram()

        #Encoder
        self.conv1 = torch.nn.Conv2d(1, 4, 3, padding=1)  
        self.conv2 = torch.nn.Conv2d(4, 8, 3, padding=1)
        # self.conv3 = torch.nn.Conv2d(128, 128, 4, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = x[:,:,0]
        x2 = x[:,:,1]

        # Left channel encoder
        x1 = self.spec(x1)
        x1 = torch.reshape(x1, (x1.shape[0], 1, x1.shape[1], x1.shape[2]))
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool(x1)
        # x1 = F.relu(self.conv3(x1))


        # Right channel encoder
        x2 = self.spec(x2)
        x2 = torch.reshape(x2, (x2.shape[0], 1, x2.shape[1], x2.shape[2]))
        x2 = F.relu(self.conv1(x2))
        x2 = self.pool(x2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.pool(x2)
        # x2 = F.relu(self.conv3(x2))


        x = torch.cat((x1,x2), dim=1)     
        return x

def register_models():
    register_custom_encoder('vizdoom', VizdoomEncoder)
    register_custom_encoder('vizdoomSound', VizdoomSoundEncoder)
