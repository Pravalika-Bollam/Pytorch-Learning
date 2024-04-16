# imports
import torch
import torch.nn as nn # linear, cnn
import torch.optim as optim # sgd, adam
import torchvision.datasets as datasets #standard datasets
import torchvision.transforms as transforms # transformations on data
from torch.utils.data import DataLoader # create mini batches, handle data
import torch.nn.functional as F # relu, tanh



# create fully connected network

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,  num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,  hidden_size , num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size ).to(device = device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# hyperparamters
input_size = 28
hidden_size  = 256
sequence_length = 28
num_layers = 2
num_classes = 10
batch_size = 60
epochs = 2
learning_rate = 0.002

# load data

train_dataset = datasets.MNIST(root = 'dataset/', train = True, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size , shuffle = True)

#initialize network

model = RNN(input_size ,  hidden_size, num_layers, num_classes).to(device= device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network

for epoch in range(epochs):
    print(f"epoch {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device = device).squeeze(1)
        target = target.to(device = device)
        scores = model(data)
        loss = criterion(scores, target)
        #backward prop
        optimizer.zero_grad() #not store from previous for props
        loss.backward()
        #gradient or adam step
        optimizer.step()
        
        
    
# accuracy 
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Train Dataset")
    else:
        print('Test Dataset')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device).squeeze(1)
            y = y.to(device = device)
            scores = model(x)
            _, predictions = scores.max(1) # we want index not value
            num_correct += (predictions == y).sum()  
            num_samples += predictions.size(0)
        print(f'got {num_correct} out of {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model )