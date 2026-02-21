import torch
import  torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#compose applies transformations in order
#toTensor converts to tensor (3, 32, 32)
#normalize normalizes the image, original pixel values are between 0 and 255, we want between -1 and 1


batch_size = 4 # 4 images per batch

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
#num_workers is the number of subprocesses to use for data loading

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


#the difference between trainset and testset is that trainset is used for training and testset is used for testing

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")




def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy() #convert to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #it makes the image more readable
    plt.show()
    
dataiter = iter(trainloader) # iter returns an iterator object that you can use to iterate over the data
images, labels = next(dataiter) #next returns the next item in the iterator

#show images
# imshow(torchvision.utils.make_grid(images)) # imshow shows a grid of images

#print labels 
#print(" ".join(f'{classes[labels[j]]:5s}' for j in range(batch_size))) 


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
       #What do we put in the constructor? 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
        self.conv2 =nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        #an affine operation is a linear combination of inputs and weightsy = Wx+b
        self.fc1 = nn.Linear (in_features=16*5*5, out_features=120) # 16 is the number of output channels
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        #fc1 is the first fully connected layer
        #fc2 is the second fully connected layer 
        #fc3 is the third fully connected layer
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 is the first convolutional layer
        x = self.pool(F.relu(self.conv2(x))) # conv2 is the second convolutional layer
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x)) #
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x #x is the output 
        
net = Network()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2): # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.
            
print('Finished Training')
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)