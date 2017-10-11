# Simple GAN network on MNIST data

This project is intended to work as a first simple example of implementing
a GAN network. A GAN network is a deep learning generative algorithm, in which
two networks - a generator and a discriminator - compete against each other in
order to create, in this case, images. The generator tries to create fake images,
while the discriminator tries to distinguish between generated and real images.

## Getting started

These instructions will get a copy of the project up and running on your computer

### Prerequisites

In order to run the program, you will need Tensorflow, Matplotlib and numpy.
The program was written and works in Python 3.6.2; for other versions, it is
uncertain whether or not the software works.

Tensorflow:

```
pip install tensorflow
```

Matplotlib:

```
pip install matplotlib
```

numpy:

```
pip install numpy
```

### Running the program

In your terminal, simply run

```
python gan_mnist.py
```

If you want an untrained network, delete the content in the checkpoints folder!

### Acknowledgements

Special thanks to [Augustinus Kristiadus' blog on GANs in tensorflow](https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/) for having provided both theory and code for implementing a GAN network in tensorflow.
