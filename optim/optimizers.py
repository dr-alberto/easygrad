from easygrad.tensor import Tensor


class Optimizer():
  def __init__(self, params):
    self.params = [param for param in params if param.require_grad]

  def zero_grad(self):
    for param in self.params:
      param.grad = None

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr

  def step(self):
   for p in self.params:
    p -= p.grad * self.lr

class RMSProp(Optimizer):
  def __init__(self, params, lr=0.001, gamma=0.9, eps=1e-6):
    super().__init__(params)
    self.lr, self.gamma, self.eps = lr, gamma, eps
    self.v = [Tensor.zeros(*t.shape()) for t in params]

  def step(self):
    for i, t in enumerate(self.params):
      self.v[i] = self.gamma * self.v[i] + (1.0 - self.gamma) * t.grad * t.grad
      t -= (t.grad * self.lr).div(self.v[i].sqrt() + self.eps)
