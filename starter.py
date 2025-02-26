import librosa
import torch 
import torch.nn as nn
import sklearn
import numpy as np
from itertools import pairwise
from object_storage import object_open
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
# import mir_eval

# use GPU if available, otherwise, use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VGGishPlusMLP(nn.Module):
    def __init__(self, finetuning: bool, mlp_hidden_dimensions: tuple = ()):
        super().__init__()
        self.vggish = VGGISH.get_model()
        for param in self.vggish.parameters():
            param.requires_grad = finetuning

        in_dims = (128,) + mlp_hidden_dimensions
        self.mlp = torch.nn.Sequential()
        for in_dim, out_dim in pairwise(in_dims):
            self.mlp.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(in_features=in_dims[-1], out_features=1))


    def forward(self, x):
        return self.mlp(self.vggish(x))

def load_audio(audio_path): #load audio and return signal and sample rate
    y, sr = librosa.load(audio_path)
    return y, sr

def logmel_spectogram(audio_path): #returns log mel spectorgram
    y, sr = load_audio(audio_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, win_length=2048, hop_length=320)
    log_mel = librosa.power_to_db(mel, ref=np.max) #log mel spectrogram
    return log_mel

def vggish_melspectrogram(audio_path):
    melspec_proc = VGGISH.get_input_processor()
    waveform, original_rate = torchaudio.load(object_open(audio_path, 'rb'))
    waveform = waveform.squeeze(0)
    waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
    melspec = melspec_proc(waveform)
    return melspec


# def train():
#     return 0

# def evaluate(): #mir eval library
#     return 0

# def encoder_input(audio_path): 
#     return 0 #return input for encoder


def beatTracker(audio_path):
    y, sr = load_audio(audio_path)
    model = VGGishPlusMLP(finetuning=True, mlp_hidden_dimensions=(64,)).to(device)
    return beats, downbeats #return vector of beat times in seconds + downbeats

if __name__ == '__main__':
    inputfile = '/Users/marikaitiprimenta/Desktop/Music Informatics/cw1/BallroomData/ChaChaCha/Albums-Cafe_Paradiso-05.wav'
    beats, downbeats = beatTracker(inputfile)
    print(beats)
    print(downbeats)