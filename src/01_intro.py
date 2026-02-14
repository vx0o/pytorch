import torch
import numpy as np

print(torch.__version__)


##Initialize a random tensor
data  = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print ("x_data:", x_data)

np_array = np.array(data) 
x_np = torch.from_numpy(np_array)

print ("x_np:", x_np)

x_ones = torch.ones_like(x_data)
print ("Ones Tensor:", x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print ("Random Tensor:", x_rand)

shape = (2,3,)
rand_tensor = torch.rand(shape)
print ("Random Tensor:", rand_tensor)
ones_tensor = torch.ones(shape)
print ("Ones Tensor:", ones_tensor)
zeros_tensor = torch.zeros(shape)
print ("Zeros Tensor:", zeros_tensor)


hundred_zeros = torch.zeros(100, 100)
print("100x100 zeros tensors:", hundred_zeros)

#Attributes of a tensor
tensor1 = torch.rand(1, 2)
print(tensor1.storage())
print(f"Shape of tensor : {tensor1.shape}")
print(f"Datatype of tensor : {tensor1.dtype}")
print(f"Device tensor is stored on : {tensor1.device}")

tensor2 = torch.ones(4, 4)
print(f"First row: {tensor2[0]}")


#SLICING AND INDEXING
x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
tensor3 = torch.tensor(x)
print(tensor3)

print(tensor3[0]) #First row
print(tensor3[:, 0]) #First column

#: means all the rows, 0 means the first column. So this will give us the first column of the tensor.

print(tensor3[0:2, :]) #First 2 rows and all the columns
print(tensor3[:, 0:2]) #All the rows and first 2 columns
print(tensor3[1:, :2]) #All the rows from index 1 and first 2 columns

# start is inclusive, end is exclusive. So 0:2 means from index 0 to index 1 (2 is exclusive). So this will give us the first 2 rows and all the columns.

#Joining tensors
tensor4 = torch.tensor([[1, 2], [3, 4]])
tensor5 = torch.tensor([[5, 6], [7, 8]])

t1 = torch.cat([tensor4, tensor5], dim=0)
print("Concatenated along rows (dim=0):", t1)

tensor = torch.cat([tensor4, tensor5], dim=1)
print("Concatenated along columns (dim=1):", tensor)

t2 = torch.stack([tensor4, tensor5], dim=0)
print("Stacked along new dimension (dim=0):", t2)

x = torch.tensor([1, 1, 1, 1, 1,])
# it's equal to torch.ones(5)

print(x.shape)
concat = torch.cat([x, x], dim=0)
print("Concatenated 1D tensors:", concat)

y = torch.tensor([1, 2, 3])
z = torch.tensor([4, 5, 6])
concat2 = torch.cat([y, z], dim=0)
print("Concatenated 1D tensors:", concat2)

# Dimension out of range (expected to be in range of [-1, 0], but got 1)

