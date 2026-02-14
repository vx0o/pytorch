import torch

# Tensor creation
x = torch.ones(5)
y = torch.tensor([1, 1, 1, 1, 1])

print(x)
print(y)

# Shape and dimensions
print(x.shape)
print(x.ndim)
print(x.numel())

# Indexing
z = torch.tensor([[1,2,3],[4,5,6]])
print(z[0])
print(z[:,1])

# Concatenation
a = torch.ones(1,5)
b = torch.ones(1,5)

concat = torch.cat([a,b], dim=0)
print(concat)
