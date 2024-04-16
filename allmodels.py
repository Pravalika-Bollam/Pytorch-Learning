import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.module):
    def __init__(self, in_channels, num_classes):
        super(CNN. self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, kernel_size = 3, padding = 1, stride = 1, out_channels= 8)
        self.pool = nn.MaxPool2d(stride = 2, kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 8, kernel_size = 5, padding = 1, stride = 1, out_channels = 16)
        self.fc1 = nn.Linear(16*7*7, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# for rnns, x = x.to(device).squeeze(1) we want 64*28*28 instead of 64*1*28*28
# expects the data as batch_size, sequence_length, input_size

class RNN(nn.module):
    def __init__(self,input_size, hidden_size, num_layers, num_ classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc= nn.Linear(hidden_size*sequence_length, num_classes)
    def forward(self, x):
        h0 = torch.tensors(self.num_layers, x.size(0), self.hidden_size).to(device = device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class GRU(nn.module):
    def __init__(self,input_size, hidden_size, num_layers, num_ classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.fc= nn.Linear(hidden_size*sequence_length, num_classes)
    def forward(self, x):
        h0 = torch.tensors(self.num_layers, x.size(0), self.hidden_size).to(device = device)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class LSTM(nn.module):
    def __init__(self,input_size, hidden_size, num_layers, num_ classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc= nn.Linear(hidden_size , num_classes)
    def forward(self, x):
        h0 = torch.tensors(self.num_layers, x.size(0), self.hidden_size).to(device = device)
        c0 = torch.tensors(self.num_layers, x.size(0), self.hidden_size).to(device = device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class BLSTM(nn.module):
    def __init__(self,input_size, hidden_size, num_layers, num_ classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.blstm = nn.BLSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc= nn.Linear(hidden_size*2 , num_classes)
    def forward(self, x):
        h0 = torch.tensors(self.num_layers*2, x.size(0), self.hidden_size).to(device = device)
        c0 = torch.tensors(self.num_layers*2, x.size(0), self.hidden_size).to(device = device)
        out, _ = self.blstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out








device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)

input_size = 784
num_classes = 10
batch_size = 64
learning_rate = 0.02
epochs = 3
in_channels = 1


train_dataset = datasets.MNIST(root = 'dataset/', train = True, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, download = True, transform = transforms.ToTensor())
test_loader = DataLoader(dataset = test_dataset, shuffle = True, batch_size = batch_size)


model = NN(input_size = input_size, num_classes = num_classes).to(device = device)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device = device)
        target = target.to(device = device)
        score = model(data)
        loss = criterion(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('train dataset')
    else:
        print('test dataset')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device= device)
            y = y.to(device = device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == target).sum()
            num_samples += preds.size(0)
    print(f'accuracy is {float(num_correct)/ float(num_samples)*100:.2f}')
    
            