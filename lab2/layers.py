import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        self.name = 'Linear'
        
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.bias = np.zeros((1, out_dim))
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.bias
    
    def backward(self, grad_out, lr=0.01):
        grad_input = np.dot(grad_out, self.W.T)
        grad_W = np.dot(self.input.T, grad_out)
        grad_bias = np.sum(grad_out, axis = 0, keepdims=True)

        self.dX = grad_input
        self.dW = grad_W
        self.db = grad_bias
        
        self.W -= lr * grad_W
        self.bias -= lr * grad_bias
        
        return grad_input
    

class ReLu:
    def __init__(self):
        self.name = 'ReLu'
    
    def forward(self, x):
        self.input = x
        return np.maximum(x, 0)
    
    def backward(self, out_grad):
        self.dX = out_grad * (self.input > 0)
        return self.dX
    
    
class Softmax:
    def __init__(self):
        self.name = 'Softmax'
    
    def forward(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def backward(seld, out_grad):
        return out_grad


class MSELoss:
    def __init__(self):
        self.name = 'MSELoss'
    
    def forward(self, pred, gt):
        self.pred = pred
        self.gt = gt
        return np.mean((pred - gt)**2)
    
    def backward(self):
        return 2 * (self.pred - self.gt) / self.pred.size