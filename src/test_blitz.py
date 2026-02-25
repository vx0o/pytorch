import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

PATH  = "./cifar_net.pth"

transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4 # 4 images per batch

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy() #convert to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #it makes the image more readable
    plt.show()



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
net.load_state_dict(torch.load(PATH, weights_only=True))
net.eval()

print(net)

dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


output = net(images)
_, predicted = torch.max(output, 1)
print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# per-class accuracy
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')