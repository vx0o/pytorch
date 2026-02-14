import torch
print(torch.__version__)
x  = torch.rand(2, 3)


print("Shape:", x.shape)
print("Size:", x.size())