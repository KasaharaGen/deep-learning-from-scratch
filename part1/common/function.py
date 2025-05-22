import numpy as py

def identity_function(x):
    return x

def step_function(x):
    return np.array(x >0, dtype=int)

def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y

def sigmoid_grad(x):
    y_grad = (1.0 - sigmoid(x)) * sigmoid(x)
    return y_grad

def relu(x):
    y = np.maximum(0,x)
    return y

def relu_grad(x):
    y_grad = np.zeros_like(x)
    y_grad[x>0] = 1
    return y_grad

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) #オーバーフロー対策
    y = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdim=True)
    return y

def sum_squared_error(y,t):
    loss = 0.5 * np.sum((y-t)**2)
    return loss

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7) / batch_size

    return loss

def softmax_loss(X,t):
    y = softmax(X)
    loss = cross_entropy_error(y,t)
    return loss
