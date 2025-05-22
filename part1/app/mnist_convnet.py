import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.pardir)  # ../ をパスに追加
from common.trainer import Trainer
from common.simple_convnet import SimpleConvNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# MNISTデータの取得と整形
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int64)
X = X.reshape(-1, 1, 28, 28) / 255.0  # 正規化

# 学習/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ネットワークとトレーナーの初期化
max_epochs = 20
model = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01
)

trainer = Trainer(
    model, X_train, y_train, X_test, y_test,
    epochs=max_epochs,
    mini_batch_size=100,
    optimizer='Adam',
    optimizer_param={'lr': 0.001},
    evaluate_sample_num_per_epoch=1000
)

# 学習の実行
trainer.train()

# モデルパラメータ保存
model.save_params("params.pkl")
print("Saved Network Parameters!")

# 精度ログ保存
log_data = {
    'train_acc_list': trainer.train_acc_list,
    'test_acc_list': trainer.test_acc_list
}
with open("accuracy_log.pkl", "wb") as f:
    pickle.dump(log_data, f)
print("Saved Accuracy Logs!")


# 精度ログの読み込み
with open("accuracy_log.pkl", 'rb') as f:
    data = pickle.load(f)

train_acc_list = data['train_acc_list']
test_acc_list = data['test_acc_list']

# プロット
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()
