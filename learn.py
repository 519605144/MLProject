import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def make_plot(X, y, plot_name):
    plt.figure(figsize=(12, 8))
    plt.title(plot_name, fontsize=30)
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()

make_plot(X, y, "Classification Dataset Visualization ")

class Layer():
       def __init__(self, n_input, n_output, activation=None, weights=None, bias=None):
              """
              :param n_input: 输入点数目
              :param n_output:输出点数目
              :param activation: 激活函数
              :param weights: 权值
              :param bias: 偏值
              """
              self.weights = weights if weights is None else np.random.randn(n_input, n_output)* np.sqrt(1 / n_output)
              self.bias = bias if bias is None else np.random.randn(n_output) * 0.1
              self.activation = activation
              self.activation_output = None
              self.error = None
              self.delta = None

       def activate(self, X):
              r = np.dot(X, self.weights) + self.bias #输出
              self.activation_output = self._apply_activation(r)
              return self.activation_output

       def _apply_activation(self, r):
              if self.activation is None:
                     return r
              elif self.activation == 'relu':
                     return np.max(r, 0)
              elif self.activation == 'signoid':
                     return 1/(1+np.exp(-r))
              elif self.activation == 'tanh':
                     return np.tanh(r)

              return r

       def apply_activation_derivative(self, r):
              if self.activation is None:
                     return np.ones_like(r)
              elif self.activation=='relu':
                     grad = np.array(r, copy=True)
                     grad[r>0] = 1
                     grad[r<=0] = 0
                     return grad
              elif self.activation=='signoid':
                     return  r*(1-r)
              elif self.activation=='tanh':
                     return 1-r**2
              

