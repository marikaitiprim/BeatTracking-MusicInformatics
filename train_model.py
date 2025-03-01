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

# use GPU if available, otherwise, use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VGGishfinetune(nn.Module):
    """VGGish model with the last layer unfrozen for fine-tuning."""
    def __init__(self):
        super().__init__()

        self.vggish = VGGISH.get_model()    #get the pretrained vggish model
        for param in self.vggish.parameters():  #freeze the layers
            param.requires_grad = False

        for param in list(self.vggish.parameters())[-2:]:   #unfreeze the last layer - weights and biases
            param.requires_grad = True

    def forward(self, x):
        return self.vggish(x)
    

def evaluate(model, data_loader, criterion):
    model.eval()
    num_batches = len(data_loader)
    epoch_loss = 0.
    accuracy = 0.
    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_outputs = model(batch_inputs).squeeze(dim=1)
            batch_binary_outputs = torch.where(batch_outputs < 0.5, 0, 1)
            accuracy += (batch_binary_outputs == batch_labels).sum().item()
            epoch_loss += criterion(batch_outputs, batch_labels).item()
    epoch_loss /= num_batches
    accuracy /= len(data_loader.dataset)
    return epoch_loss, accuracy


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_model, evaluate_every_n_epochs=1):
    model.train()
    num_batches = len(train_loader)
    best_valid_acc = 0.0
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # forward + backward + optimize
            outputs = model(batch_inputs).squeeze(dim=1)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        epoch_loss /= num_batches
        # print training loss
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        train_losses.append(epoch_loss)
        
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
    return train_losses, valid_losses, valid_accuracies



if __name__ == '__main__':

    audio_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData"
    annotation_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master"

    train_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=32)

    

    # for mel_spec, beats in train_loader:
    #     print(mel_spec.shape)  # (batch_size, 1, 2117, 64)
    #     print(beats.shape)     # (batch_size, 2117)
    #     break