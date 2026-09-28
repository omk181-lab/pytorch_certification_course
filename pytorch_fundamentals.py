import torch
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

print(torch.__version__)


## Introduction to Tensors 

# Creating tensors 

# Scalar 

scalar = torch.tensor(7)

print(scalar)

## pytoech tensors are created using torch.tensor() funtion. 

print(scalar.ndim)

## Get tensor back as Python int 
print(scalar.item())