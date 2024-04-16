# imports
import torch
import torch.nn as nn # linear, cnn
import torch.optim as optim # sgd, adam
import torchvision.datasets as datasets #standard datasets
import torchvision.transforms as transforms # transformations on data
from torch.utils.data import DataLoader # create mini batches, handle data
import torch.nn.functional as F # relu, tanh

# create fully connected network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# hyperparamters
input_size = 784
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

model = NN(input_size = input_size, num_classes = num_classes).to(device= device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device = device)
        target = target.to(device = device)
        data = data.reshape(data.shape[0], -1)
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
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1) # we want index not value
            num_correct += (predictions == y).sum()  
            num_samples += predictions.size(0)
        print(f'got {num_correct} out of {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model )
    
            
        
            
            
            
        
    






