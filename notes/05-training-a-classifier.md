## TRAINING A CLASSIFIER
---
### DATA 
When we deal with any kind of data ( image, text, audio, video ) we usually use standard python packages that load data into a numpy array and rhen you can convert this array into a torch.*Tensor 

For convinience in this example we are using torchvision, that has dataloaders for common datasets such as  ImageNet, CIFAR10, MNIST etc. and data transformers for images, viz., torchvision.datasets and torch,utils.data.DataLoader

Here we are using the CIFAR10 dataset. The images are of size 3x32x32 

---

## STEPS

1. Load and normalise the CIFAR10 training and test datasets using torchvision
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

### 1. Load and normalise CIFAR10

We loaded the CIFAR dataset using torchvision.datasets.CIFAR10

it has a dataset of 60000 colour images of size 32x32 pixels, divided  into 10 classes.

The dataset is plit  between 50000 training images and 10000 testing images

Defining a transform:
transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


This applies two teansformations:

transforms.ToTensor()
- COnverts th eimage into a pytorch tensor, changes the image shape, converts the pixel value from 0-255, to 0.0-1.0

transform.Normalize()
- Normalizes the tensor value to range -1  to 1, which improves the training stability and performance, neural networks train faster when inputs are normalised

Tehn we created the training and testing datasets
- train=True loads the training set
- train=False loads the testing set

We used the DataLoader to load images in batches
Each batch has shape (4, 3, 32, 32)

This prepares the dataset to be fed into the neural network.