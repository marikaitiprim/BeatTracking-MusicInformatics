import os
import librosa
import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
import load_data
import test_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

# use GPU if available, otherwise, use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VGGishfinetune(nn.Module):
    """VGGish model with the last layer unfrozen for fine-tuning."""
    def __init__(self, mlp_hidden_dimensions: tuple = ()):
        super().__init__()

        self.vggish = VGGISH.get_model()    #get the pretrained vggish model
        for param in self.vggish.parameters():  #freeze the layers
            param.requires_grad = False

        for param in list(self.vggish.parameters())[-2:]:   #unfreeze the last layer - weights and biases
            param.requires_grad = True

        in_dims = (128,) + mlp_hidden_dimensions
        self.mlp = torch.nn.Sequential()
        for in_dim, out_dim in zip(in_dims[0:-1], mlp_hidden_dimensions):
            self.mlp.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(in_features=in_dims[-1], out_features=64)) #time steps output

    def forward(self, x):
        return self.mlp(self.vggish(x))
    

def evaluate(model, data_loader, criterion):
    model.eval()
    num_batches = len(data_loader)
    epoch_loss = 0.
    accuracy = 0.
    total_beats = 0
    # all_labels = []
    # all_predictions = []
    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            batch_size, num_setofframes, num_channels, mel_bins, time_step = batch_inputs.shape
            batch_inputs = batch_inputs.reshape(batch_size*num_setofframes, num_channels, mel_bins, time_step) #reshape to (batch_size*num_setofframes, num_channels, mel_bins, time_step)
            batch_labels = batch_labels.reshape(batch_size*num_setofframes, time_step) #reshape to (batch_size*num_setofframes, time_step)

            batch_outputs = model(batch_inputs) #(960,64)

            batch_binary_outputs = torch.where(batch_outputs < 0.5, 0, 1)

            accuracy += ((batch_binary_outputs == batch_labels) & (batch_binary_outputs == 1)).sum().item()
            total_beats += (batch_labels == 1).sum().item() #total number of beats in the batch
            epoch_loss += criterion(batch_outputs, batch_labels).item()

            # all_labels.extend(batch_labels.cpu().numpy().flatten())
            # all_predictions.extend(batch_binary_outputs.cpu().numpy().flatten())
            # import pdb; pdb.set_trace()
    epoch_loss /= num_batches
    # f1 = f1_score(batch_labels, batch_binary_outputs, average='macro')
    accuracy /= total_beats #divide all the correct predictions by the total number of frame predictions
    return epoch_loss, accuracy


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_model, evaluate_every_n_epochs=1):
    model.train()
    num_batches = len(train_loader)
    best_valid_acc = 0.0
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0.
        total_beats = 0
        for batch_inputs, batch_labels in tqdm(train_loader):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            batch_size, num_setofframes, num_channels, mel_bins, time_step = batch_inputs.shape
            batch_inputs = batch_inputs.reshape(batch_size*num_setofframes, num_channels, mel_bins, time_step) #reshape to (batch_size*num_setofframes, num_channels, mel_bins, time_step)
            batch_labels = batch_labels.reshape(batch_size*num_setofframes, time_step) #reshape to (batch_size*num_setofframes, time_step)
            
            # forward + backward + optimize
            outputs = model(batch_inputs)        #squeeze removes the channel dimension
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()
            outputs_binary = torch.where(outputs < 0.5, 0, 1)
            epoch_accuracy += ((outputs_binary == batch_labels) & (outputs_binary == 1)).sum().item()
            total_beats += (batch_labels == 1).sum().item() #total number of beats in the batch

        epoch_loss /= num_batches
        epoch_accuracy /= total_beats #divide all the correct predictions by the total number of frame predictions
        # print training loss
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        print(f'[{epoch+1}] accuracy: {100*epoch_accuracy:.2f}%')
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # evaluate the network on the validation data
        if((epoch+1) % evaluate_every_n_epochs == 0):
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
            print(f'Validation loss: {valid_loss:.6f}')
            print(f'Validation accuracy: {100*valid_acc:.2f}%')
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)
            
            # if the best validation performance so far, save the network to file 
            if(valid_acc >= best_valid_acc):
                best_valid_acc = valid_acc
                print('Saving best model')
                torch.save(model.state_dict(), saved_model)
    return train_losses, valid_losses, train_accuracies, valid_accuracies

def plot_metrics(train_losses, valid_losses, valid_accuracies):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_tight_layout(True)
    ax1.plot(train_losses, label='Training')
    ax1.plot(valid_losses, label='Validation')
    ax1.set_xlabel('epochs')
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(valid_accuracies)
    ax2.set_xlabel('epochs')
    ax2.set_title('Validation Accuracy')
    plt.show()

if __name__ == '__main__':

    audio_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData"
    annotation_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master"

    train_loader, test_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=32)

    model = VGGishfinetune().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, valid_losses, train_accuracies, valid_accuracies = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, saved_model='best_model.pth')
    plot_metrics(train_losses, valid_losses, valid_accuracies)

    # # test the model
    # best_model_path = '/Users/marikaitiprimenta/Desktop/best_model.pth'
    # model.load_state_dict(torch.load(best_model_path))
    # model.eval()
    # test_audio = '/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData/ChaChaCha/Albums-Cafe_Paradiso-05.wav'
    # test_annotation = '/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master/Albums-Cafe_Paradiso-05.beats'

    # test_dataset = load_data.BeatDataset([test_audio], [test_annotation])
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    # test_loss, test_acc = evaluate(model, test_loader, criterion)
 
    # print(f'Test loss: {test_loss:.6f}')
    # print(f'Test accuracy: {100*test_acc:.2f}%')
