import torch
from numpy import random as np

# # Tensor creation
# x = torch.ones(5)
# y = torch.tensor([1, 1, 1, 1, 1])

# print(x)
# print(y)

# # Shape and dimensions
# print(x.shape)
# print(x.ndim)
# print(x.numel())

# # Indexing
# z = torch.tensor([[1,2,3],[4,5,6]])
# print(z[0])
# print(z[:,1])

# # Concatenation
# a = torch.ones(1,5)
# b = torch.ones(1,5)

# concat = torch.cat([a,b], dim=0)
# print(concat)

#multiplying tensors
# tensor1 = torch.tensor([[1, 2], [3, 4]])
# tensor2 = torch.tensor([[5, 6], [7, 8]])

# tensor3 = tensor1 * tensor2 # or tensor1.mul(tensor2)
# print(tensor3)

#NumPy Array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t : {t}")
print(f"n : {n}")