<p align="center">
 <h1 align="center">Face-Generator</h1>
</p>

## Introduction

GAN (Generative Adversarial Network) consists of a Generator creating realistic data and a Discriminator distinguishing real from fake. 
They compete adversarially, enabling realistic data generation for images, text, and more.

## Dataset

 The dataset which I used is CelebA, that is human face dataset and you can dowload it from google easily 
 
## Demo
<p align="center">
  <img src="image/output.gif" width=650><br/>
  <i>Demo</i>
</p>

## Training

To train the model, run the train.py with the command:
python train.py
the file trained_model will be automatically created and the model will be saved through epochs in file last.pt

## Requirements

* **python**
* **cv2**
* **pytorch** 
* **numpy**
* **matplotlib**
