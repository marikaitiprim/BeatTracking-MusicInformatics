import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchaudio.prototype.pipelines import VGGISH
import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np


# 1st architecture tested
# class VGGishfinetune(nn.Module): 
#     """VGGish model with the last layer unfrozen for fine-tuning."""
#     def __init__(self, mlp_hidden_dimensions: tuple = ()):
#         super().__init__()

#         self.vggish = VGGISH.get_model()    #get the pretrained vggish model
#         for param in self.vggish.parameters():  #freeze the layers
#             param.requires_grad = False

#         for param in list(self.vggish.parameters())[-2:]:   #unfreeze the last layer - weights and biases
#             param.requires_grad = True

#         in_dims = (128,) + mlp_hidden_dimensions
#         self.mlp = torch.nn.Sequential()
#         for in_dim, out_dim in zip(in_dims[0:-1], mlp_hidden_dimensions):
#             self.mlp.append(nn.Linear(in_features=in_dim, out_features=out_dim))
#             self.mlp.append(nn.ReLU())
#         self.mlp.append(nn.Linear(in_features=in_dims[-1], out_features=64)) #time steps output

#     def forward(self, x):
#         return self.mlp(self.vggish(x))


# 2nd architecture tested
# class CNNModel(nn.Module):
#     def __init__(self, num_classes=64):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(128 * 12 * 8, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.7)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x) 
        # return x

#3rd architecture tested
# class VGGishfinetune_lstm(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=64):
#         super().__init__()

#         self.vggish = VGGISH.get_model()    #get the pretrained vggish model
#         for param in self.vggish.parameters():  #freeze the layers
#             param.requires_grad = False

#         for param in list(self.vggish.parameters())[-2:]:   #unfreeze the last layer - weights and biases
#             param.requires_grad = True

#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, 64)  # Bidirectional LSTM

#     def forward(self, x):
#         x = self.vggish(x)
#         x, _ = self.lstm(x)
#         x = self.fc(x)
#         return x
    
class CNNBeatTracker(nn.Module):
    def __init__(self, num_classes=64):
        super(CNNBeatTracker, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128 * 12 * 8, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) #flatten

        x, _ = self.lstm(x)

        x = self.fc2(x)
        
        return x
    

def peak_picking(batch_outputs, device):        #function for post-processing. Peak picking for converting the probabilities of the output of the model into binary representation
    batch_outputs_cpu = batch_outputs.cpu().numpy()
    batch_binary_outputs = np.zeros(batch_outputs_cpu.shape)  # Initialize with all zeros

    for frame_set_id in range(batch_outputs_cpu.shape[0]): #for every frame set in the batch
        detected_peaks_indices, _ = scipy.signal.find_peaks(batch_outputs_cpu[frame_set_id, :], distance=30) #post-processing peak picking for extracting the beats of the set
        batch_binary_outputs[frame_set_id, detected_peaks_indices] = 1  # Set beats to 1 for the specific set

    return torch.tensor(batch_binary_outputs, dtype=torch.float32).to(device)
    
def calculate_precision(true_positives, false_positives):
    precision = 0.
    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    return precision

def calculate_recall(true_positives, false_negatives):
    recall = 0.
    if true_positives + false_negatives != 0:
        recall = true_positives / (true_positives + false_negatives)
    return recall

def calculate_f1(precision, recall):
    f1_accuracy = 0.
    if precision + recall != 0:
        f1_accuracy = 2*precision*recall / (precision + recall)
    return f1_accuracy

def pos_weight_loss(data_loader):
    num_positive = 0
    num_negative = 0

    for _, batch_labels in data_loader:  # Extract only labels and divide 
        num_positive += (batch_labels == 1).sum().item()
        num_negative += (batch_labels == 0).sum().item()

    if num_positive > 0:  
        return torch.tensor([num_negative / num_positive], dtype=torch.float32).to(device)
    else:
        return torch.tensor([1.0], dtype=torch.float32).to(device)  # Default to 1 if no positives

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

            batch_binary_outputs = peak_picking(batch_outputs=batch_outputs, device=device)    #post-processing
        
            true_positives += ((batch_binary_outputs == batch_labels) & (batch_binary_outputs == 1)).sum().item() #All beat predictions - TP
            false_positives += ((batch_binary_outputs != batch_labels) & (batch_binary_outputs == 1)).sum().item() #FP
            false_negatives += ((batch_binary_outputs != batch_labels) & (batch_binary_outputs == 0)).sum().item() #FN
            epoch_loss += criterion(batch_outputs, batch_labels).item()
           
    epoch_loss /= num_batches

    precision = calculate_precision(true_positives=true_positives, false_positives=false_positives)
    recall = calculate_recall(true_positives=true_positives, false_negatives=false_negatives)
    f1_accuracy = calculate_f1(precision=precision, recall=recall)

    return epoch_loss, precision, recall, f1_accuracy

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_model, evaluate_every_n_epochs=1):
    model.train()
    num_batches = len(train_loader)
    best_valid_acc = 0.0
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    val_criterion =  nn.BCEWithLogitsLoss(pos_weight=pos_weight_loss(valid_loader)) #different pos_weight for validation set

    for epoch in range(num_epochs):
        epoch_loss = 0

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

        epoch_loss /= num_batches

        scheduler.step()

        # print training loss
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        train_losses.append(epoch_loss)
        
        # evaluate the network on the validation data
        if((epoch+1) % evaluate_every_n_epochs == 0):
            valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, val_criterion)
            print(f'Validation loss: {valid_loss:.6f}')
            print(f'Validation Precision: {100*valid_precision:.2f}% | Recall: {100*valid_recall:.2f}% | F1: {100*valid_f1:.2f}%')
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_f1)
            
            # if the best validation performance so far, save the network to file 
            if(valid_f1 >= best_valid_acc):
                best_valid_acc = valid_f1
                print('Saving best model')
                torch.save(model.state_dict(), saved_model)
    return train_losses, valid_losses, valid_accuracies

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
    plt.savefig("best_model_metrics.png")

if __name__ == '__main__':

    # use GPU if available, otherwise, use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using ", device , ":")

    audio_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomData"  #change to the path of your audio data
    annotation_dir = "/Users/marikaitiprimenta/Desktop/Beat-Tracking---Music-Informatics/BallroomAnnotations-master" #change to the path of your annotation data

    train_loader, test_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=32) 

    model = CNNBeatTracker().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_loss(train_loader))  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs


    train_losses, valid_losses, valid_accuracies = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, saved_model='best_model.pth')
    plot_metrics(train_losses, valid_losses, valid_accuracies)
