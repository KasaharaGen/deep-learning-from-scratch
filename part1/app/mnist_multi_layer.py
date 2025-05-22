import sys,os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from common.multi_layer_net import MultiLayerNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int64)

train_loss_list = []
train_acc_list = []
test_acc_list = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

iters_num = 10000
train_size = X_train.shape[0]
batch_size = 100
learning_rate = 0.01

iter_per_epoch = max(train_size / batch_size, 1)

model = MultiLayerNet(input_size=784,hidden_size_list=[512,256,128,64,32],output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    grad = model.gradient(X_batch,y_batch)

    for key in ('W1','b1','W2','b2'):
        model.params[key] -= learning_rate * grad[key]

    loss = model.loss(X_batch,y_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))  # エポック数に合わせたx軸

plt.plot(x, train_acc_list, label='train acc', marker=markers['train'])
plt.plot(x, test_acc_list, label='test acc', marker=markers['test'], linestyle='--')

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



