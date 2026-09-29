import torch
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
print(torch.__version__)


# Introduction to Tensors 

Creating tensors 

## Scalar  LOWER CASE VARIABLE 

scalar = torch.tensor(7)

print(scalar)

## pytorch tensors are created using torch.tensor() funtion. 

print(scalar.ndim)

## Get tensor back as Python int 
print(scalar.item())

## Vector  is a 1 dimension LOWER CASE VARIABLE
vector = torch.tensor([7,7])
print(vector)
print(f'vector dimensions:{vector.ndim}')
print(f'vector shape:{vector.shape}')
print(f'vector type:{vector.type()}')

## MATRIX  UPPPER CASE VARIABLE 

MATRIX = torch.tensor([[1,2,3],
                       [3,4,5],
                       [6,7,8]])

print(MATRIX)
print(f'MATRIX dimensions: {MATRIX.ndim}')
print(f'MATRIX shape : {MATRIX.shape}')
print(f'MTRIX type : {MATRIX.type()}')

print(MATRIX[0])


## TENSOR  UPPER CASE VARIABLE 

TENSOR = torch.tensor([[[1,2,3],
                        [3,4,5],
                        [6,7,8]],
                        [[3,3,3],
                        [3,4,5],
                        [6,7,8]]])
print(f'TENSOR : {TENSOR}')

print(f'TENSOR dimensions : {TENSOR.ndim}')
print(f'TENSOR shape : {TENSOR.shape}')
print(f'TENSOR type : {TENSOR.type()}')


print(TENSOR[1][0], TENSOR[0][1])

## RANDOM TENSORS 

Why random tensors? 
Random tensors are important because the way neural network works is that they start with tensors full of random numbers 
and then adjust those random numberss to better represent the data. 

Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers 

Create a random tensor of shape (3,4)
random_tensor = torch.rand(3,4)
print(random_tensor)
print(random_tensor.ndim)

Create a radom tensor with similar shape to an image tensor 

random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, color chnnales 

print(random_image_size_tensor, random_image_size_tensor.ndim,random_image_size_tensor.shape)

plt.imshow(random_image_size_tensor.numpy())
plt.axis('off')
plt.show()

random_image_size_tensor_permuted = random_image_size_tensor.permute(2,0,1)
print(f'permuted shape : {random_image_size_tensor_permuted.shape}')


