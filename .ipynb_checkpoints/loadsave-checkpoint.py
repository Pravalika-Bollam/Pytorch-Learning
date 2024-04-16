# imports
import torch
import torch.nn as nn # linear, cnn
import torch.optim as optim # sgd, adam
import torchvision.datasets as datasets #standard datasets
import torchvision.transforms as transforms # transformations on data
from torch.utils.data import DataLoader # create mini batches, handle data
import torch.nn.functional as F # relu, tanh

# create fully connected network


class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, stride = 1, padding = 1, kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, stride = 1, padding = 1, kernel_size = 3)
        self.fc1 = nn.Linear(16*7*7, num_classes)
    def forward(self, x):
        x= F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# hyperparamters
in_channels= 1
num_classes = 10
batch_size = 60
epochs = 3
learning_rate = 0.002
load_model = True
# load data

train_dataset = datasets.MNIST(root = 'dataset/', train = True, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size , shuffle = True)

#initialize network

model = CNN(in_channels = in_channels, num_classes = num_classes).to(device= device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network
def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



for epoch in range(epochs):
    print(f'epoch is {epoch}')
    losses = []
    if epoch%2 == 0:
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device = device)
        target = target.to(device = device)
        scores = model(data)
        loss = criterion(scores, target)
        #backward prop
        optimizer.zero_grad() #not store from previous for props
        loss.backward()
        #gradient or adam step
        optimizer.step()
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    
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
            x = x.to(device = device)
            y = y.to(device = device)
            scores = model(x)
            _, predictions = scores.max(1) # we want index not value
            num_correct += (predictions == y).sum()  
            num_samples += predictions.size(0)
        print(f'got {num_correct} out of {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model )