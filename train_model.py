import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchaudio.prototype.pipelines import VGGISH
import load_data
import test_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

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
    precision = 0.
    recall = 0.
    f1_accuracy = 0.
    true_positives = 0.
    false_positives = 0.
    false_negatives = 0.
    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            batch_size, num_setofframes, num_channels, mel_bins, time_step = batch_inputs.shape
            batch_inputs = batch_inputs.reshape(batch_size*num_setofframes, num_channels, mel_bins, time_step) #reshape to (batch_size*num_setofframes, num_channels, mel_bins, time_step)
            batch_labels = batch_labels.reshape(batch_size*num_setofframes, time_step) #reshape to (batch_size*num_setofframes, time_step)

            batch_outputs = model(batch_inputs) #(960,64)

            batch_binary_outputs = torch.where(batch_outputs < 0.5, 0, 1)

            true_positives += ((batch_binary_outputs == batch_labels) & (batch_binary_outputs == 1)).sum().item() #All beat predictions - TP
            false_positives += ((batch_binary_outputs != batch_labels) & (batch_binary_outputs == 1)).sum().item() #FP
            false_negatives += ((batch_binary_outputs != batch_labels) & (batch_binary_outputs == 0)).sum().item() #FN
            epoch_loss += criterion(batch_outputs, batch_labels).item()
           
    epoch_loss /= num_batches

    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.

    if true_positives + false_negatives != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.
    
    if precision + recall != 0:
        f1_accuracy = 2*precision*recall / (precision + recall)
    else:
        f1_accuracy = 0.

    return epoch_loss, precision, recall, f1_accuracy


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
        # epoch_accuracy /= total_beats #divide all the correct predictions by the total number of frame predictions

        # print training loss
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        # print(f'[{epoch+1}] accuracy: {100*epoch_accuracy:.2f}%')
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # evaluate the network on the validation data
        if((epoch+1) % evaluate_every_n_epochs == 0):
            valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion)
            print(f'Validation loss: {valid_loss:.6f}')
            print(f'Validation Precision: {100*valid_precision:.2f}% | Recall: {100*valid_recall:.2f}% | F1: {100*valid_f1:.2f}%')
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_f1)
            
            # if the best validation performance so far, save the network to file 
            if(valid_f1 >= best_valid_acc):
                best_valid_acc = valid_f1
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

    # use GPU if available, otherwise, use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using ", device , ":")

    audio_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData"
    annotation_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master"

    train_loader, test_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=32)

    num_positive = 0
    num_negative = 0

    for _, batch_labels in train_loader:  # Extract only labels and divide 
        num_positive += (batch_labels == 1).sum().item()
        num_negative += (batch_labels == 0).sum().item()

    if num_positive > 0:  
        pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32).to(device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)  # Default to 1 if no positives

    model = VGGishfinetune().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #overfitting with 0.01 (trainloss goes down - validationloss goes up) - 0.001 default

    train_losses, valid_losses, train_accuracies, valid_accuracies = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, saved_model='best_model.pth')
    plot_metrics(train_losses, valid_losses, valid_accuracies)

    # test the model
    # best_model_path = '/Users/marikaitiprimenta/Desktop/best_model.pth'
    # model = VGGishfinetune().to(device)
    # model.load_state_dict(torch.load(best_model_path))
    # model.eval()
    # test_audio = '/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData/ChaChaCha/Albums-Cafe_Paradiso-05.wav'
    # test_annotation = '/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master/Albums-Cafe_Paradiso-05.beats'

    # test_dataset = load_data.BeatDataset([test_audio], [test_annotation])
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


    # num_positive = 0
    # num_negative = 0

    # for _, batch_labels in test_loader:  # Extract only labels and divide 
    #     num_positive += (batch_labels == 1).sum().item()
    #     num_negative += (batch_labels == 0).sum().item()

    # if num_positive > 0:  
    #     pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32).to(device)
    # else:
    #     pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)  # Default to 1 if no positives

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # test_loss, _,_,test_acc = evaluate(model, test_loader, criterion)
 
    # print(f'Test loss: {test_loss:.6f}')
    # print(f'Test accuracy: {100*test_acc:.2f}%')
