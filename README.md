# ディープラーニングをスクラッチから実装

## 概要
本プロジェクトは研究の一環として、深層学習技術をスクラッチから実装することを目的としている。  
書籍「ゼロから作るディープラーニング」シリーズ（オライリー・ジャパン）をベースに、NumPyによる基礎実装から、抽象フレームワークの構築まで体系的に学習・実装を行った。

---

## 特徴

- 数式からアルゴリズムまでを一貫して**NumPyで再構成**
- 実装を通じて、**誤差逆伝播・自動微分・Attention・演算グラフ**などの動作原理を体得


---

## リポジトリ構成

### part1：[./part1](./part1)
- 書籍：「ゼロから作るDeep Learning ❶」
- 内容：NumPyによるDNN／CNNの実装、MNIST分類
- 特徴：誤差逆伝播、勾配検証、活性化関数、最適化アルゴリズムをすべてスクラッチで実装

### part2：[./part2](./part2)
- 書籍：「ゼロから作るDeep Learning ❷」
- 内容：単語分散表現（word2vec）、RNN、Seq2Seq、Attention
- 特徴：自然言語処理に必要な系列データ処理と中間表現の獲得方法を学習

### part3：[./part3](./part3)
- 書籍：「ゼロから作るDeep Learning ❸」
- 内容：PyTorch/TensorFlow風のミニDLフレームワークをNumPyで模倣
- 特徴：`Variable`, `Function`, `Trainer` などを抽象的に構成し、動的な演算グラフ・自動微分の理解を深める

---

## 成果例

| モデル | 内容 |
|--------|------|
| DNN | MNIST分類（精度98%以上） |
| CNN | conv → pooling → flatten → softmax での画像認識 |
| word2vec | 単語の類似度可視化（cos類似度＋t-SNE） |
| Seq2Seq | 単語レベルのエンコーダ・デコーダ構築 |
| DeZero | 逆伝播可能な演算グラフによるMLP学習（自作） |

---

## 実行方法

```bash
# Python仮想環境推奨
python -m venv venv
source venv/bin/activate

プロジェクトはAnaconda環境を使用

# 依存ライブラリ（NumPy, matplotlib など）
pip install -r requirements.txt

# 実行例（part1）
cd part1
python train_mnist.py
