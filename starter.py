import librosa
import torch 
import torch.nn as nn
import sklearn
import numpy as np
# import mir_eval
from transformers import Wav2Vec2Processor, Wav2Vec2Model 

# Load pretrained Wav2Vec2 model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

class Encoder(nn.Module):       #encoder architecture
    def __init__(self, embed_dim=1024, num_layers=8, num_heads=8, ff_dim=1024, dropout=0.1):
        """
        Transformer Encoder similar to Wav2Vec2 and HuBERT.
        
        Args:
            embed_dim (int): Dimension of embeddings.
            num_layers (int): Number of Transformer encoder layers.
            num_heads (int): Number of attention heads.
            ff_dim (int): Hidden dimension of feed-forward network.
            dropout (float): Dropout rate.
        """

        super().__init__()
        self.wav2vec2 = wav2vec2_model  # Pretrained Wav2Vec2 model
        encoder_layer = nn.TransformerEncoderLayer( #“Attention Is All You Need”. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. paper
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",  # Typically used in Wav2Vec2 -> checkkkkk
            batch_first=True     # Ensures (batch, seq, feature) input format
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) #stack of transformer encoder layers

    def forward(self, x):
        """
        Forward pass through the Transformer Encoder.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            
        Returns:
            Tensor: Encoded representations (batch_size, seq_len, embed_dim).
        """

        # Process the raw audio with Wav2Vec2 (Feature extraction)
        input_values = processor(x, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        features = self.wav2vec2(input_values).last_hidden_state
        return self.encoder(features)
    

def load_audio(audio_path): #load audio and return signal and sample rate
    y, sr = librosa.load(audio_path)
    return y, sr

def logmel_spectogram(audio_path): #returns log mel spectorgram
    y, sr = load_audio(audio_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, win_length=2048, hop_length=320)
    log_mel = librosa.power_to_db(mel, ref=np.max) #log mel spectrogram
    return log_mel

# def NTXentLoss(): #loss function chosen by the paper 
#     return 0


# def train():
#     return 0


# def evaluate(): #mir eval library
#     return 0

# def encoder_input(audio_path): 
#     return 0 #return input for encoder


def beatTracker(audio_path):
    y, sr = load_audio(audio_path)
    model = Encoder()
    output = model(y)
    return output
    # return beats, downbeats #return vector of beat times in seconds + downbeats

if __name__ == '__main__':
    inputfile = '/Users/marikaitiprimenta/Desktop/Music Informatics/cw1/BallroomData/ChaChaCha/Albums-Cafe_Paradiso-05.wav'
    print(beatTracker(inputfile))
    # beats, downbeats = beatTracker(inputfile)
    # print(beats)
    # print(downbeats)