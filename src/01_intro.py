import torch
import numpy as np

print(torch.__version__)


##Initialize a random tensor
# data  = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)

# print ("x_data:", x_data)

# np_array = np.array(data) 
# x_np = torch.from_numpy(np_array)

# print ("x_np:", x_np)

# x_ones = torch.ones_like(x_data)
# print ("Ones Tensor:", x_ones)

# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print ("Random Tensor:", x_rand)

# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# print ("Random Tensor:", rand_tensor)
# ones_tensor = torch.ones(shape)
# print ("Ones Tensor:", ones_tensor)
# zeros_tensor = torch.zeros(shape)
# print ("Zeros Tensor:", zeros_tensor)


# hundred_zeros = torch.zeros(100, 100)
# print("100x100 zeros tensors:", hundred_zeros)

##Attributes of a 
# tensor1 = torch.rand(2, 4)
# print(f"Shape of tensor : {tensor1.shape}")
# print(f"Datatype of tensor : {tensor1.dtype}")
# print(f"Device tensor is stored on : {tensor1.device}")

tensor2 = torch.ones(4, 4)
print(f"First row: {tensor2[0]}")

