import os
import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.prototype.pipelines import VGGISH

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


class BeatDataset(Dataset):
    def __init__(self, audio_paths, annotation_paths):
        self.audio_paths, self.annotation_paths = audio_paths, annotation_paths

    def load_annotations(self, annotation_path):
        annot = np.loadtxt(annotation_path)
        return torch.tensor(annot, dtype=torch.float32)

    def vggish_melspectrogram(self, audio_path):    #returns vggish melspectrogram
        melspec_proc = VGGISH.get_input_processor()
        waveform, original_rate = torchaudio.load(audio_path)

        start_time = 0
        end_time = 29   #10 seconds
        waveform = waveform[:, int(start_time*original_rate):int(end_time*original_rate)]   #extract only the 10-20 seconds of the audio

        waveform = waveform.squeeze(0)
        waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
        melspec = melspec_proc(waveform) #(num_setofframes, 96, 64) 

        melspec = torch.cat([melspec], dim=0)  # Shape: (num_setofframes, 1, 96, 64)

        return melspec

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """Returns (mel spectrogram, beat annotations) pair."""
        audio_path = self.audio_paths[idx]
        annotation_path = self.annotation_paths[idx]

        mel_spec = self.vggish_melspectrogram(audio_path)
        annotations = self.load_annotations(annotation_path)

        beats = torch.zeros(mel_spec.shape[0]*mel_spec.shape[3])   #matrix of zeros of shape (num_setofframes, 64)
        for beat_time in annotations[:, 0]:
            beat_time_samples = beat_time * VGGISH.sample_rate
            for frame_index in range(mel_spec.shape[0]* mel_spec.shape[3]):
                if frame_index*160 + 400 > beat_time_samples and frame_index*160 <= beat_time_samples: #hop_size = 160 and frame_size = 400 for VGGish
                    beats[frame_index] = 1

        return mel_spec, beats 


def load_data(audio_dir, annotation_dir, batch_size=32):

    # Create dataset
    audio_paths, annotation_paths = create_dataset(audio_dir, annotation_dir)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(audio_paths, annotation_paths, test_size=0.2, random_state=42)

    # Create training and test datasets
    train_dataset = BeatDataset(X_train, y_train)
    test_dataset = BeatDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

