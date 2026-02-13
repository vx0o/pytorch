import torch
print(torch.__version__)
print(torch.rand(2, 3))

print("Shape:", torch.rand(2, 3).shape)
print("Size:", torch.rand(2, 3).size())