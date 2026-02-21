# Neural Networks — PyTorch Notes

## Overview

- Neural networks can be constructed using `torch.nn`
- `torch.nn` depends on autograd so it can define models and compute gradients automatically
- `nn.Module` contains layers and a method `forward(input)` that returns an output

A neural network in Python is a class that contains layers and a forward function.

clas Network = AI Brain


- Convolutional layers → detect patterns  
- Linear layers → make decisions  

---

## Training Procedure

Typical training procedure for a neural network:

- Define the neural network with learnable parameters (weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss
- Propagate gradients back into the parameters
- Update the weights of the network using a rule such as: weight = weight - learning_rate × gradient


---

## Defining the Network

Layers are created in the constructor using statements like:

```python
self.conv1 = nn.Conv2d(...)
self.fc1 = nn.Linear(...)```

A convolution layer example:
```python
self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
```

## What Outputs Represent

Outputs from convolutional layers represent different feature detectors.

Each filter learns to detect a specific pattern in the image.

Example:

- Filter 1 → detects edges
- Filter 2 → detects corners
- Filter 3 → detects curves
- Filter 4 → detects textures

Each filter produces its own feature map.

More filters = more types of features detected.

---

## Pooling Layer

Pooling reduces the size of the image while keeping important information.

Example: 28x28 --> 14x14


Purpose:

- Reduce computation
- Keep important features
- Remove unnecessary detail
- Improve efficiency

Max pooling selects the maximum value from each region.

---

## Activation Function (ReLU)

ReLU (Rectified Linear Unit) removes negative values.

Example: [-3, 5, -1, 2] --> [0, 5, 0, 2]


Purpose:

- Adds non-linearity
- Helps the network learn complex patterns

Important rule:

Do NOT use ReLU on the final layer because the loss function applies the correct normalization.

---

## Network Flow (Conceptually)

A neural network works like a function:
input --> layer --> layer --> output



Each layer transforms the data into a more useful representation.

---

## Forward Pass

The `forward()` function defines how data flows through the network.

Flow example:
input image --> conv layer --> relu activation --> pooling  --> linear layers --> output


This produces the prediction.

---

## Backward Pass — Explanation

net.zero_grad()

- Clears all previous gradients
- Required because PyTorch accumulates gradients by default
- Without this, gradients would stack on top of previous ones

out.backward(torch.randn(1, 10))

- Computes gradients using autograd
- The tensor passed to backward() must match the output shape

- Example output shape:

- (1, 10)

- So backward must use:

- (1, 10)

- After calling backward(), gradients are stored in each parameter’s .grad attribute.

- These gradients are later used by the optimizer to update weights. 

---

## Key Operations Summary

ReLU (Rectified Linear Unit)

- Removes negative values
- Adds non-linearity
- Helps the network learn complex patterns

- Example:

- [-3, 5, -1, 2] → [0, 5, 0, 2]

---

Max Pool

- Reduces image size
- Keeps important features
- Improves efficiency
- Reduces computation

- Example:

- 28×28 → 14×14

---

Flatten

- Converts multi-dimensional image data into a vector
- Required before fully connected (Linear) layers

- Example:

- (12, 5, 5) → 300

---

Optimizer

- Updates weights automatically using gradients
- Makes the network improve over time

- Update rule:

- weight = weight − learning_rate × gradient

- Example:

- optimizer.step()

---

## Core PyTorch Components

torch.Tensor

- Multi-dimensional array
- Stores numerical data
- Supports autograd operations like backward()
- Stores gradients in .grad

---

nn.Module

- Base class for neural networks
- Contains layers and parameters
- Used to build models
- Provides forward pass functionality

Example:

class Network(nn.Module)

---

nn.Parameter

- Special Tensor automatically registered as a trainable parameter
- Updated during training
- Used for weights and biases

---

autograd.Function

- Implements forward and backward operations
- Every Tensor operation creates a Function node
- Tracks computation history
- Enables automatic gradient calculation
- Forms the computation graph
