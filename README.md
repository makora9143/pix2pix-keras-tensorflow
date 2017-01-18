# pix2pix-keras-tensorflow

Keras and TensorFlow implementation of [Image-to-Image Translation Using Conditional Adversarial Networks][https://arxiv.org/pdf/1611.07004v1.pdf] that learns a mapping from input images to output images.

## Setup

### Prerequistic

- Software
    - python2
    - tensorflow 0.12
    - keras 
    - numpy
    - scipy
    - matplotlib
    - progressbar2

- Hardware
    - nVIDIA GPU (Highly Recommend) 

### How to start (WIP)

- Clone this repo to your PC.

```bash
$ git clone https://github.com/makora9143/pix2pix-keras-tensorflow.git
$ cd pix2pix-keras-tensorflow

```
- To train the model, just run the command below. (It will takes few hours.)

```bash
$ python train.py [dataset]

```
The generated sample images is in the `output_imgs` directory.
If you want to generate some images, run this command:

```bash
$ python test.py
```

