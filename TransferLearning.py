import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



# hyperparamters
in_channels = 3
num_classes = 10
batch_size = 60
epochs = 3
learning_rate = 0.0015

# load data

train_dataset = datasets.CIFAR10(root = 'dataset/', train = True, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.CIFAR10(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size , shuffle = True)

#initialize network

model = torchvision.models.vgg16(weights='DEFAULT')
model.avgpool = nn.Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, 10))


# Freeze the weights of the pre-trained layers
for param in model.features.parameters():
    param.requires_grad = False

# For the additional layers
for param in model.classifier.parameters():
    param.requires_grad = True

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network

for epoch in range(epochs):
    print(f'epoch is {epoch}')
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
    


