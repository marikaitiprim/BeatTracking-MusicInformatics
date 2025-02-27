import librosa
import torch 
import torch.nn as nn
import sklearn
import numpy as np
# from itertools import pairwise
# from object_storage import object_open
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
# import mir_eval

# use GPU if available, otherwise, use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VGGishPlusMLP(nn.Module):
    def __init__(self, mlp_hidden_dimensions: tuple = ()):
        super().__init__()

        self.vggish = VGGISH.get_model()    #get the pretrained vggish model
        for param in self.vggish.parameters():  #freeze the layers
            param.requires_grad = False

        for param in list(self.vggish.parameters())[-2:]:   #unfreeze the last layer - weights and biases
            param.requires_grad = True

        in_dims = (128,) + mlp_hidden_dimensions
        self.mlp = torch.nn.Sequential()
        for in_dim, out_dim in zip(in_dims[:-1], in_dims[1:]):
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

# def vggish_melspectrogram(audio_path):
#     melspec_proc = VGGISH.get_input_processor()
#     waveform, original_rate = torchaudio.load(object_open(audio_path, 'rb'))
#     waveform = waveform.squeeze(0)
#     waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
#     melspec = melspec_proc(waveform)
#     return melspec


def train(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        current_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item() * inputs.size(0)
        
        epoch_loss = current_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

# def evaluate(): #mir eval library
#     return 0

# def test():     #test the model 
    # return 0


def beatTracker(audio_path):
    y, sr = load_audio(audio_path)
    model = VGGishPlusMLP().to(device) #frozen layers
    # Example usage
    # model = VGGishPlusMLP(finetuning=True, mlp_hidden_dimensions=(128, 64, 32)).to(device)
    # criterion = nn.MSELoss()  # or another suitable loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Assuming you have a DataLoader `train_loader` for your training data
    # train_model(model, train_loader, criterion, optimizer, num_epochs=25)
    return beats, downbeats #return vector of beat times in seconds + downbeats

if __name__ == '__main__':
    inputfile = '/Users/marikaitiprimenta/Desktop/Music Informatics/cw1/BallroomData/ChaChaCha/Albums-Cafe_Paradiso-05.wav'
    beats, downbeats = beatTracker(inputfile)
    print(beats)
    print(downbeats)