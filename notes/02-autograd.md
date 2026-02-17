
# PyTorch Autograd Notes

## What is Autograd?

Autograd is PyTorch’s automatic differentiation system. It automatically calculates gradients (derivatives) of tensors with respect to a loss function. These gradients are used to train neural networks.

In simple terms, autograd tells the model how to adjust its weights to reduce error.

---

## requires_grad

The `requires_grad` flag tells PyTorch whether to track operations on a tensor.

Example:

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)

This means PyTorch will track all operations involving x.

If requires_grad=False, PyTorch will ignore the tensor during gradient computation.

Forward Pass

The forward pass is when the model performs calculations to produce an output.

Example:

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
print(Q)


Output:

tensor([-12., 65.], grad_fn=<SubBackward0>)


grad_fn shows that PyTorch is tracking how this tensor was created.

Computation Graph (DAG)

PyTorch builds a computation graph automatically when operations are performed on tensors with requires_grad=True.

Example flow:

a, b → operations → Q


DAG = Directed Acyclic Graph
This graph allows PyTorch to compute gradients during backpropagation.

Backward Pass

The backward pass calculates gradients of the output with respect to inputs.

Example:

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)


Gradients are stored in:

print(a.grad)
print(b.grad)


Output:

tensor([36., 81.])
tensor([-12., -8.])


These values show how much each input affects the output.

Mathematically:

dQ/da = 9a²
dQ/db = -2b

Accessing Gradients

Gradients are stored in the .grad attribute.

Example:

print(a.grad)


Gradients are used by the optimizer to update weights.

grad_fn

Example:

tensor([-12., 65.], grad_fn=<SubBackward0>)


grad_fn tells which operation created the tensor.

Examples:

AddBackward0
SubBackward0
MulBackward0
PowBackward0


This is part of the computation graph.

Gradient Tracking Rules

If no inputs require gradients:

x = torch.rand(5,5)
y = torch.rand(5,5)
z = x + y

print(z.requires_grad)


Output:

False


If at least one input requires gradients:

z = torch.rand(5,5, requires_grad=True)
b = x + z

print(b.requires_grad)


Output:

True


Rule: If any input requires gradients, the output will also require gradients.

Freezing Parameters

Freezing prevents parameters from being updated during training.

Example:

for param in model.parameters():
    param.requires_grad = False


This is useful when finetuning pretrained models.

Classifier Layer

The classifier is the final layer that converts features into predictions.

Example:

model.fc = nn.Linear(512, 10)


This replaces the classifier to output 10 classes instead of 1000.

Only unfrozen layers will be trained.

Optimizer and Gradient Descent

Example:

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.step()


The optimizer updates parameters using gradients stored in .grad.

Formula:

new_weight = old_weight − learning_rate × gradient

Training Step Summary

A full training step:

prediction = model(data)
loss = (prediction - labels).sum()

loss.backward()

optimizer.step()


Steps:

Forward pass → model makes prediction

Loss calculation → measures error

Backward pass → calculates gradients

Optimizer step → updates weights