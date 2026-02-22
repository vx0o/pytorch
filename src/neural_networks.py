#defining a network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
         #What do we put in the constructor? 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # a convolution is a linear combination of the pixels in a window of the image
        # conv2d takes in the number of input channels, the number of output channels, and the size of the kernel
        # kernel size is the size of the window that the convolution is applied to
        
        self.conv2 =nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        #an affine operation is a linear combination of inputs and weightsy = Wx+b
        self.fc1 = nn.Linear (in_features=12*5*5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        #fc1 is the first fully connected layer
        #fc2 is the second fully connected layer 
        #fc3 is the third fully connected layer
        
    def forward(self, input):
        c1 = F.relu(self.conv1(input)) # conv1 is the first convolutional layer
        #relu is the rectified linear unit activation function
        #c1 is the output of the first convolutional layer
        #input is the input to the first convolutional layer
        #outputs a tensor of the same shape as the input, N is the batch size
        #Tensor with size (N, 6, 28, 28)
        
        s2 = F.max_pool2d(c1, (2,2))
        #subsampling , 2x2 grid , does not ahve any parameters and outputs a tensor (N, 6, 14, 14)
        
        c3  = F.relu(self.conv2(s2))
        #6 input channels and 16 output channels, 5x5 square convolutuion, outputs a (N, 12, 10, 10) Tenrsor
        
        s4 = F.max_pool2d(c3, (2,2))
        #2x2 grid, outputs a temsor of (N, 12, 5, 5)
     
        s4= torch.flatten(s4, 1)
        #flattens the tensor and outputs a tensor of (N, 300)
        
        f5 = F.relu(self.fc1(s4))
        #fully connected layer Tensor input is (N, 400)and out put is (N, 120)
        
        f6 = F.relu(self.fc2(f5))
        #fully connected layer Tensor input is (N, 120)and out put is (N, 84)
        
        output = self.fc3(f6)
        #fully connected layer Tensor input is (N, 84)and out put is (N, 10)
        
        return output 





net  = Network()

optimizer = optim.SGD(net.parameters(), lr=0.01)

criterion = nn.MSELoss()
print(net)   
#         #before moving on explain what we are going to do
#         #1. Define the layers
#         #2. Define the loss function
#         #3. Define the optimizer
        
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

input = torch.randn(1, 1, 32, 32)
#size of batch is 1, 1 input channel, 32x32 image
target = torch.rand(10) #a test traget
target = target.view(1, -1) # make it the same shape as output
output = net(input)
print(output)  
loss = criterion(output, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss)


print(loss.grad_fn) #mseloss
print(loss.grad_fn.next_functions[0][0]) #linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # relu

