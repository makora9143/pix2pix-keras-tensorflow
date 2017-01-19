# pix2pix-keras-tensorflow

Keras and TensorFlow hybrid-implementation of [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf) that learns a mapping from input images to output images.
This implementation is as same as possible to the original paper.

The examples from the paper:
![examples](original.jpg)


## Setup

### Prerequistic

- Software
    - python2.7
    - tensorflow==0.12.0
    - keras==1.2.0 
    - numpy==1.11.3
    - scipy==0.18.1
    - matplotlib==1.5.3
    - progressbar2==3.12.0

- Hardware
    - nVIDIA GPU (Highly Recommend) 

### Install

- Clone this repo to your PC.

```bash
$ git clone https://github.com/makora9143/pix2pix-keras-tensorflow.git
$ cd pix2pix-keras-tensorflow

```

### Usage (WIP)

- To train the model, just run the command below. (It will takes few hours.)
  - [dataset] = facades / cityscapes / maps / edges2shoes / edges2handbags
```bash
$ python train.py -d [dataset]

```
- The generated sample images is in the `output_imgs` directory.
If you want to generate some images, run this command:

```bash
$ python test.py
```


# pix2pix-keras-tensorflow

画像から出力画像への変換を学習する[Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)のKerasとTensorflowを組み合わせた実装です．
可能な限り，論文内及び著者の実装に準拠しています．

元論文の出力例：
![examples](original.jpg)


## 設定

### 必要な環境

- ソフトウェア・ライブラリ
    - python2.7
    - tensorflow==0.12.0
    - keras==1.2.0 
    - numpy==1.11.3
    - scipy==0.18.1
    - matplotlib==1.5.3
    - progressbar2==3.12.0

- ハードウェア
    - nVIDIA GPU (推奨) 

### 準備

- ローカルPCに`git clone`してください．

```bash
$ git clone https://github.com/makora9143/pix2pix-keras-tensorflow.git
$ cd pix2pix-keras-tensorflow


### 使い方


```
- 学習するために，次のコマンドを実行してください．(数時間かかります．)
  - [データセット] = facades / cityscapes / maps / edges2shoes / edges2handbags
```bash
$ python train.py -d [データセット]

```
- 生成された画像は，`output_imgs`ディレクトリに出力されます．
画像を生成するために，次のコマンドを実行します．

```bash
$ python test.py
```
