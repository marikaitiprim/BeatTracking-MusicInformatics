import torch
import load_data
import train_model
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.prototype.pipelines import VGGISH
import IPython.display as ipd

def vggish_melspectrogram(audio_path):    #returns vggish melspectrogram - adapted for testing
    melspec_proc = VGGISH.get_input_processor()
    waveform, original_rate = torchaudio.load(audio_path)

    waveform = waveform.squeeze(0)
    waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
    melspec = melspec_proc(waveform) #(num_setofframes, 96, 64) 

    melspec = torch.cat([melspec], dim=0)  # Shape: (num_setofframes, 1, 96, 64)

    return melspec


def beatTracker(audio_path):

    best_model_path = '/Users/marikaitiprimenta/Desktop/best_model.pth' #path for the pretrained model
    model = train_model.VGGishfinetune().to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    xtest = vggish_melspectrogram(audio_path)

    with torch.no_grad():
        xtest = xtest.to(device)
        output = model(xtest)
        
        binary_output = torch.where(output < 0.5, 0, 1)

        #convert to seconds to extract the beats
        # beats = 


    return binary_output.cpu().numpy(), downbeats

def plot_beats(audio_path, beats):

    melspec_proc = VGGISH.get_input_processor()
    waveform, _ = torchaudio.load(audio_path)

    plt.figure(figsize=(14, 3))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.plot(waveform, 'c')

    plt.title('Beat tracking')
    for beat in beats:
        plt.axvline(beat, ymin=0.5, color='g')

    ipd.display(ipd.Audio(audio_path))


if __name__ == "__main__":


    # use GPU if available, otherwise, use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using ", device , ":")

    inputFile = "path"
    beats, downbeats = beatTracker(inputFile)
    plot_beats(inputFile, beats)

