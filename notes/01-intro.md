01 - Intro



\##Notes



\#Inittiliazing a tensor



Tensors are speciliased data structure, like matrices

we use them to encode the inputs and outputs of a model and its parameters



they can run on GPUs and other hardware accelerators.

They can share the same memory with NumPy, so theres no use to copy data



they can be created directly from data \[], form NumPy arrays (.from\_numpy())or from another tensor



the new tensoer retains the properties of the older tensor (shape, datatype) unless explixity overridden



ones\_like creates a tensor of ones, witht he same data type and size as the one we passed in as a parameter

rand\_like create a tensor of random numbers with the same data type and size as the one we passed in as parameter,

if we explicitly state to change the data type the returned values are going to be of the data type we declared



torch.rand\_like()



.tensor() creates tensor based on  the data we give it, .ones\_like, rand\_like, \_like, creates a tensor with the dimension.



\#Attributes of a tensor

Describe their shape, datatype and the device on which they are sorted



\##Operations on Tensors

They perform over 1200 operations including arithmetic linear algebra matrixes etc.

they can be run on gpus and accelerators  but by default they are created on the cpu

we move them using .to method



Exploring and experimenting with the operations of tensors



torch.is\_tensor --> returns true if obj is tensor



.is\_storage, .storage() --> when a tensor is created  Pytorch allocates memory somewhere in RAM and that contains the amount of numbers inthe  tensor

sorage doesn't care about rows or columns is just a flat list

.is\_nonzero()

set/get\_default\_dtype()

numel() --> reutns the total number of element sin the input type

ndim() --> number of dimension

shape() --> structure

.arrange --> range of values eg x = torch.arrange(0, 10), output  \[0, 1, 2,3, 4, 5, 6, 7, 8, 9]

useul for indexing

.linespace() --> evenly spaced values

eye() --> diagonal tensor with ones in the middle ansd 0s elsewhere

full() --> retuns a tensor of size with fill\_value





\##SLICING PATTERNS:

x\[0]        → first row

x\[-1]       → last row

x\[:, 0]     → first column

x\[:, -1]    → last column

x\[:2]       → first 2 rows

x\[:, :2]    → first 2 columns

x\[:, :]     → entire tensor



\## Tensor Dimensions



\### Definition

The number of dimensions (`ndim`) of a tensor is the number of axes in its shape.



You can check it using:



```python

tensor.shape

tensor.ndim



