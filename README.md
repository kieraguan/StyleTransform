# GYZ: The Universal Style Transfer via Feature Transforms

Neural Networks and Deep Learning ECBM E4040 - Fall 2018 Student Project, Columbia University

This is a reproduction of the NIPS'17 paper [Universal Style Transfer via Feature Transforms](http://papers.nips.cc/paper/6642-universal-style-transfer-via-feature-transforms)

## Group Member

- Tingyu Guan  tg2663
- Hong Yan    hy2557
- Yimeng Zhang yz3397

## Requirements

- Python 3.6.5
- tensorflow 1.8.0
- scikit-image

The project also depends on the pretrained model parameters file [vgg19.npy](https://github.com/tensorlayer/pretrained-models/raw/master/models/vgg19.npy). Place the model npy file in the project directory.

## Dataset

The project uses MSCOCO dataset [train2014 dataset](http://images.cocodataset.org/zips/train2014.zip), After downloading the dataset, unzip the dataset outside the project directory.

## Test

1. Download the [vgg19.npy](https://github.com/tensorlayer/pretrained-models/raw/master/models/vgg19.npy) and put it **inside** the project directory
2. Download the [train2014 dataset](http://images.cocodataset.org/zips/train2014.zip) and unzip it **outside** the project directory
3. Run the *image_style_transfer.ipynb* to test all the modules and to see the result of the project.
4. Modify and run the autotrain.py to train a module.

## Module Explanation

- autotrain.py       Automatically train a decoder
- encoder.py        The tensorflow network of the image encoder
- decoder.py        The tensorflow network of the image decoder
- wct.py           A numpy implementatino of whitening color transformer
- utils.py          Functions of loading images
- decoder_trainer.py   The tensorflow network used to train the decoder
- style_transferrer.py  THe tensorflow network combining the encoder, wct and the decoder to stylize an image.