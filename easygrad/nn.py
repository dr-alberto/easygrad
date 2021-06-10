# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
from tensor import Tensor
import numpy as np
from math import *

# Loss functions 

def cross_entropy(y_pred, y_true):
  return -sum([y_true[i]*log2(y_pred[i]) for i in range(len(y_pred))]) 

def binary_classification(y_pred, y_true):    
  return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

# ----


if __name__ == "__main__":
  predictions = np.array([[0.25,0.25,0.25,0.25],
                          [0.01,0.01,0.01,0.96]])
  targets = np.array([[0,0,0,1], [0,0,0,1]])
  ans = 0.71355817782  #Correct answer
  x = cross_entropy(predictions, targets)
  print(np.isclose(x,ans))

  p = [0.10, 0.40, 0.50]
  q = [0.80, 0.15, 0.05]

  print(cross_entropy(q, p))
  


