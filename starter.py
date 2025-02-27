import os
import librosa
import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
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

def create_dataset(audio_dir, annotation_dir):      #pair paths from audio and annotations
    audio_paths = []
    annotation_paths = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                annotation_path = os.path.join(annotation_dir, file.replace('.wav', '.beats'))
                if os.path.exists(annotation_path):
                    audio_paths.append(audio_path)
                    annotation_paths.append(annotation_path)

    return audio_paths, annotation_paths

def load_audio(audio_path): #load audio and return signal and sample rate
    y, sr = librosa.load(audio_path)
    return y, sr

def logmel_spectrogram(audio_path, fixed_len=2117): #returns log mel spectorgram
    y, sr = load_audio(audio_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, win_length=2048, hop_length=320)
    log_mel = librosa.power_to_db(mel, ref=np.max) #log mel spectrogram

    if log_mel.shape[1] < fixed_len:
        log_mel = np.pad(log_mel, ((0, 0), (0, fixed_len - log_mel.shape[1])), 'constant')
    else:
        log_mel = log_mel[:, :fixed_len]

    return log_mel

def vggish_melspectrogram(audio_path, fixed_len=2117):    #returns vggish melspectrogram
    melspec_proc = VGGISH.get_input_processor()
    waveform, original_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze(0)
    waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
    melspec = melspec_proc(waveform)

    melspec = torch.cat([melspec], dim=0)  # Shape: (num_frames, 96, 64)

    if melspec.shape[0] < fixed_len:
        melspec = torch.nn.functional.pad(melspec, (0, 0, 0, 0, 0, fixed_len - melspec.shape[0]), 'constant')
    else:
        melspec = melspec[:fixed_len, :,:]
    return melspec


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


def trainBeatTracker(audio_dir, annotations_dir, fixed_len=2117):    #train the model into the train - audio_lib 

    audio_paths, annotation_paths = create_dataset(audio_dir, annotations_dir)  #pair paths from audio and annotations

    specs = []
    annotations = []
    for audio, annotation in zip(audio_paths, annotation_paths):
        specs.append(vggish_melspectrogram(audio))                  #extract log mel spectrogram
        annot = np.loadtxt(annotation)
        if len(annot) < fixed_len:
            annot = np.pad(annot, (0, fixed_len - len(annot)), 'constant')
        else:
            annot = annot[:fixed_len]
        annotations.append(annot)              #load annotations
    
    specs = torch.stack(specs)
    annotations = torch.tensors(annotations, dtype=torch.float32)

    xtrain, xval, ytrain, yval = train_test_split(specs, annotations, test_size=0.2, random_state=42)   #split the data in 2 sets

    xtrain = torch.tensor(xtrain, dtype=torch.float32)
    ytrain = torch.tensor(ytrain, dtype=torch.float32)

    xval = torch.tensor(xval, dtype=torch.float32)
    yval = torch.tensor(yval, dtype=torch.float32)

    train_dataset = TensorDataset(xtrain, ytrain)   #create datasets
    val_dataset = TensorDataset(xval, yval)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   #create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = VGGishPlusMLP().to(device) #frozen layers
    criterion = nn.MSELoss()   #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #optimizer

    train(model, train_loader, criterion, optimizer, num_epochs=25)


def beatTracker(audio_path):    #beat tracker function for one audio file - test method
    spec = logmel_spectrogram(audio_path)    #extract log mel spectrogram

    # test()
    return beats, downbeats #return vector of beat times in seconds + downbeats

if __name__ == '__main__':
    inputfile = '/Users/marikaitiprimenta/Desktop/Music Informatics/cw1/BallroomData/ChaChaCha/Albums-Cafe_Paradiso-05.wav'
    # beats, downbeats = beatTracker(inputfile)
    # print(beats)
    # print(downbeats)

    trainBeatTracker('/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData', '/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master')
