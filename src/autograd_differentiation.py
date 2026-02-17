import torch 

#simulating one training step of a neural network
a = torch.tensor([2., 3.,], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)  #hgradients are now stored in a and b


print(9*a**2 == a.grad)
print(-2*b == b.grad) #check if collected gradients are correct

print(Q)
print(a.grad)
print(b.grad)

#note: even if only one requires_grad, all will be updated
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")