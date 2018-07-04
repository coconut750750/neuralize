# Neuralize #

## Overview ##

## Training MNIST Data ##
saving the trained network:
```bash
python3 run_mnist.py -s { filename }
```
loading a trained network:
```bash
python3 run_mnist.py -l { filename }
```
teaching a trained network even more:
```bash
python3 run_mnist.py -l { filename } -t { additional_iterations}
```
add the `-s` flag to save the new network.

## Comparing Neural Network Parameters ##
```bash
python3 compare.py
```

## Activation Functions ##