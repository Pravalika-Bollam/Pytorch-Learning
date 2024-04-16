# imports
import torch
import torchvision
import torch.nn as nn # linear, cnn
import torch.optim as optim # sgd, adam
import torchvision.datasets as datasets #standard datasets
import torchvision.transforms as transforms # transformations on data
from torch.utils.data import DataLoader # create mini batches, handle data
import torch.nn.functional as F # relu, tanh
import os
from PIL import Image

class DogsAndCatsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cat_dir = os.path.join(self.root_dir, 'Cat')
        self.dog_dir = os.path.join(self.root_dir, 'Dog')
        self.cat_files = [os.path.join(self.cat_dir, f) for f in os.listdir(self.cat_dir) if f.endswith('.png')]
        self.dog_files = [os.path.join(self.dog_dir, f) for f in os.listdir(self.dog_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.cat_files) + len(self.dog_files)

    def __getitem__(self, idx):
        if idx < len(self.cat_files):
            file = self.cat_files[idx]
            label = 0
        else:
            file = self.dog_files[idx - len(self.cat_files)]
            label = 1

        img = Image.open(file)
        if self.transform:
            img = self.transform(img)

        return img, label


dataset = DogsAndCatsDataset(root_dir = 'DeepLearning/Dog and Cat', transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))]))



#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# hyperparamters
in_channels= 3
num_classes = 2
batch_size = 20
epochs = 3
learning_rate = 0.002

train_length = int( 0.7 * len(dataset))
test_length =  len(dataset) - train_length
print(f' Train length is {train_length} \n Test length is {test_length}')
train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)



#initialize network

model = torchvision.models.resnet50(weights='DEFAULT')
num_features = model.fc.in_features 

custom_classifier = nn.Sequential(
    nn.Linear(num_features, 768),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(768, num_classes)  
)

model.fc = custom_classifier
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
    