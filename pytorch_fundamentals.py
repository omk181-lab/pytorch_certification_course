import torch
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
print(torch.__version__)


# Introduction to Tensors 

# Creating tensors 

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

# Why random tensors? 
# Random tensors are important because the way neural network works is that they start with tensors full of random numbers 
# and then adjust those random numberss to better represent the data. 

# Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers 

# Create a random tensor of shape (3,4)
random_tensor = torch.rand(3,4)
print(random_tensor)
print(random_tensor.ndim)

# Create a radom tensor with similar shape to an image tensor 

random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, color chnnales 

print(random_image_size_tensor, random_image_size_tensor.ndim,random_image_size_tensor.shape)

plt.imshow(random_image_size_tensor.numpy())
plt.axis('off')
plt.show()

random_image_size_tensor_permuted = random_image_size_tensor.permute(2,0,1)
print(f'permuted shape : {random_image_size_tensor_permuted.shape}')



### Zeros and Ones 

# Create a tensor of all zeros 

zero = torch.zeros(size=(3,3))
print(zero)
random_tensor = torch.rand(size=(3,3))
print(zero*random_tensor)

## Create a tensor of all ones 

ones = torch.ones(size=(3,4))
print(ones)

print(ones.dtype) # to get  the data type of the tensor 

## Create identity matrix  # tried one way to mask the values other than diagonal values. 

identity_matrix = torch.eye(3)
print(identity_matrix)
print(random_tensor * identity_matrix)

### Create a range of tensors and tensors-like 

# Use torch.arange() 
one_to_ten = torch.arange(start=0,end=11,step=1)
print(one_to_ten)

## Creating tensors-like 
ten_zeroes = torch.zeros_like(input=one_to_ten)
print(ten_zeroes)

one_to_eleven = torch.arange(start=1 ,end=12, step=2)
print(one_to_eleven)

six_ones = torch.ones_like(input=one_to_eleven)
print(six_ones)

## Tensor datatypes  Tensor datatype is one of the the 3 big erros with Pytorch and deep learning 
# 1. Tensors not right datatype 
# 2. Tensor not right shape 
# 3. Tensor not on the right device 

# Float32 tensor 
float_32_tensor = torch.tensor([3.0,6.0,9.0], 
                               dtype=None, # What datatype is the tensor (torch.float32 , torch.float16)
                               device=None, #  Default = CPU , Where the tensor is stored (CPU or GPU)
                               requires_grad=False) # Whether or not to track gradients with this tensor during optimization or training   ) 
print(float_32_tensor, float_32_tensor.dtype) # single precision 

# How to change tensors datatype 

float_16_tensor = float_32_tensor.type(torch.float16) # Half precision 
print(float_16_tensor)

print((float_16_tensor * float_32_tensor).dtype) 
