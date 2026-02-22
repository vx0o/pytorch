## RECAP

- torch.tensor is a multidimensional array with support for autograd operaions like backward()
- nn.Module is the neural netowrk module, it helps encapsualate parameters with helpers from moving them to the GPU, exporting loading.
- nn.PArameter is a tensor that is automatically registered as parameter when assigned as an attribute to a module
- autograd.Function implements forward and backward definitions of autograd operation. Every tensor operaton creates at least a single Function node that connect to functions that created a Tensor and encodes its history.

## LOSS FUNCTION

A loss function takes the output and target input pair and computes a value that estimates how far awat the output is from the target

Flow example:
input --> network --> output --> loss --> backward --> update weights.

---

### Step 1: Forward pass
First the input is passed throguh the network it produces an output, example shape (1,10).
These values represent the networks predictions.

---
### Step 2: Define the target

The target represents the correct answer 
target = torch.randn(10) is a test target
it MUST match the output shape so we reshape it.
target = target.view(-1, 1)

--- 
### Step 3: Define the loss function 

The loss function mesaures the error betwene output and the target
criterion = MSELoss() --> Mean Squared Error

Formula --> loss = average((output - target)^2)

If output is close to target the loss is small
if output is far from target the loss is large

---

### Step 4: Compute Loss

Use loss = criterion(output, target)
this produces a single value that represents the error

---
### Step 5: Backward pass

Compute gradients
net.zero_grad() --> clears previous gradient because gradients accumulate by default
loss.backward() --> computes gradients and stores them ub each parameters .grad attribute

Example: 
conv1.weight.grad
fc1.weight.grad
This tells the network how to adjust the weights

---

### Step 6: Update the weights

The optimizer updates the weights using gradients 

Example:
optimiser.step()

Update rule:
weight = weight - learnng_rate x gradient

This improves the network

--- 

### How loss connects to Autograd 

Autograd tracks all the operations, when loss.backward() is called autograd computes the gradients for all parameters 

this is how the network learns


---

## UPDATING THE WEIGHTS

The simplest rule suse in weight = weight - learning_rate * gradient
Implement this in pthon using: 

learning _rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

There's a vairety of different update rules.
torch.optim implemetns all these methods (code in file)


