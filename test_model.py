import torch
import train_model
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.prototype.pipelines import VGGISH
import numpy as np
import mir_eval

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
        
        binary_output = torch.where(output < 0.5, 0, 1) #convert output of the model into binary output (as done during training)

        beat_times = []
    
        for frame_set_index, beat_frames in enumerate(binary_output):
            for frame_index, is_beat in enumerate(beat_frames):

                if is_beat:
                    # Calculate the center of the frame in samples
                    frame_center_samples = ((frame_index + 64*frame_set_index) * 160) + (400 // 2)
            
                    # Convert samples to seconds
                    beat_time = frame_center_samples / VGGISH.sample_rate
                    beat_times.append(beat_time)

    return beat_times

def plot_beats(audio_path, beats_annotations, beats):

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze()
    
    total_duration = len(waveform) / sr
    time_axis = np.linspace(0, total_duration, len(waveform))

    plt.figure(figsize=(14, 6))
    plt.subplot(2,1,1)
    plt.plot(time_axis, waveform, 'c')
    print(beats_annotations)
    plt.vlines(beats_annotations, ymin=min(waveform), ymax=max(waveform), color='r', label='Beats', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Beat tracking ground truth')
    
    plt.subplot(2,1,2)
    plt.plot(time_axis, waveform, 'c')
    print(beats)
    plt.vlines(beats, ymin=min(waveform), ymax=max(waveform), color='r', label='Beats', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Beat tracking predictions')
    
    plt.tight_layout()
    plt.savefig("testing_beats.png")
    plt.show()


if __name__ == "__main__":


    # use GPU if available, otherwise, use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using ", device , ":")

    inputFile = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData/Waltz/Media-100601.wav"
    beats = beatTracker(inputFile)

    annotations = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master/Media-100601.beats"
    annot = np.loadtxt(annotations)

    plot_beats(inputFile, annot[:,0], beats)

    np.savetxt('ground_truth.txt', annot[:,0], fmt='%.6f') 
    np.savetxt('predictions.txt',beats, fmt='%.6f') 

    reference_beats = mir_eval.io.load_events('ground_truth.txt')
    estimated_beats = mir_eval.io.load_events('predictions.txt')

    # Crop out beats before 5s, a common preprocessing step
    reference_beats = mir_eval.beat.trim_beats(reference_beats)
    estimated_beats = mir_eval.beat.trim_beats(estimated_beats)

    # Compute the F-measure metric and store it in f_measure
    print("F-measure using mir_eval library: ", mir_eval.beat.f_measure(reference_beats, estimated_beats))