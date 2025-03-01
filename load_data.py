import os
import librosa
import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
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

        self.min_length = float('inf') # Determine the minimum length of the spectrograms
        for audio_path in self.audio_paths:
            spec = self.vggish_melspectrogram(audio_path)
            self.min_length = min(self.min_length, spec.shape[0])

    def load_annotations(self, annotation_path):
        annot = np.loadtxt(annotation_path)

        if len(annot) > self.min_length:
            annot = annot[:self.min_length, :]
        
        return torch.tensor(annot, dtype=torch.float32)

    def vggish_melspectrogram(self, audio_path):    #returns vggish melspectrogram
        melspec_proc = VGGISH.get_input_processor()
        waveform, original_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
        melspec = melspec_proc(waveform)

        melspec = torch.cat([melspec], dim=0)  # Shape: (num_frames, 1, 96, 64)

       # Truncate the spectrogram to match the fixed length
        if melspec.shape[0] > self.min_length:
            melspec = melspec[:self.min_length, :, :, :]

        return melspec

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """Returns (mel spectrogram, beat annotations) pair."""
        audio_path = self.audio_paths[idx]
        annotation_path = self.annotation_paths[idx]

        mel_spec = self.vggish_melspectrogram(audio_path)
        beats = self.load_annotations(annotation_path)

        return mel_spec.unsqueeze(0), beats  # Add channel dim


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

