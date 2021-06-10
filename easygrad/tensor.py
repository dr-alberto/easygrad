import numpy as np

class Tensor:
  def __init__(self, data, _children=(), op="", require_grad=True):
    self.data = self.assign_data(data)
    self.grad = 0
    self.require_grad = require_grad
    self._backward = lambda: None
    self._prev = _children

  def assign_data(self, data):
    array = np.array(data)
    return array

  def backward(self):
    topological = []
    visited = set()
    def build_topological(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topological(child)
        topological.append(v)
    build_topological(self)

    self.grad = 1
    for v in reversed(topological):
      v._backward()

  def add(self, other):
    z = Tensor(np.add(self.data, other.data), _children=(self, other))
    def _backward():
      x.grad += z.grad
      y.grad += z.grad
    z._backward = _backward
    return z

  def sum(self):
    return Tensor(sum(self.data))

  def matmul(self, other):
    return Tensor(np.matmul(self.data, other.data), _children=(self, other))

  def dot(self, other):
    return Tensor(np.dot(self.data, other.data))

  def mul(self, other):
    return Tensor(np.multiply(self.data, other.data))

  def linear(self, m=1):
    return Tensor(self.data*m)

  def relu(self):
    return Tensor(self.data * (self.data > 0))

  def leaky_relu(self, alpha=0.01):
    return Tensor(np.maximum(alpha * self.data, self.data))

  def sigmoid(self):
    return Tensor(1 / (1 + np.exp(-self.data)))
    
  def softmax(self):
    assert len(self.data.shape) == 2
    z = np.exp(self.data) / np.sum(np.exp(self.data), axis=1, keepdims=True) 
    return Tensor(z)

  def tanh(self):
    z = ((np.exp(2*self.data)-1)/(np.exp(2*self.data)+1)) 
    return Tensor(z)

  @classmethod
  def ones(self, n):
    return Tensor(np.ones(n))

  @classmethod
  def zeros(self, n):
    return Tensor(np.zeros(n))

  @classmethod
  def uniform(self, low=0, high=1, size=None):
    return Tensor(np.random.uniform(low, high, size))

  @classmethod
  def eye(self, n, m=None):
    return Tensor(np.eye(n, m))

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  def __str__(self):
    return "Tensor(%s)" % (self.data)

  

if __name__ == "__main__":
  x = Tensor([[3, 1]])
  print(x.shape, x.dtype)
