# NN-polar-decoding-equations
Implementation of neural networks for estimating the decoding equations of polar codes with arbitrary kernel size.

## Disclaimer
The repository is in constant change. The uploaded model was trained using TensorFlow 2.18.0.

## Short summary
This repository contains the code necessary to generate the training dataset and train models to approximate the decoding equations of a Polar code with arbitrary Kernel size. Feel free to play around with the training and simulation parameters to either reproduce the results in the paper or produce your own.

## Packages needed
- TensorFlow
- matplotlib
- numpy
- Optional but needed for a plug-and-play application: os, shutil, time

## To run a simulation...
1) Download everything to a folder
2) Open the generate_dataset.py file and select the dataset size, SNR, and other parameters.
3) Run to generate the dataset.
4) Open estimator_training.py, select the training SNR (must be a list that contains SNRs you have generated previously) and other parameters.
5) Run.

## Reference
The paper is currently under review for ISIT 2025.

## Authors info
Valerio Bioglio: https://scholar.google.com/citations?user=ZMGylHoAAAAJ&hl=en
Gastón De Boni Rovella: https://scholar.google.com/citations?user=ZcM9QRUAAAAJ&hl=en&authuser=1
Meryem Benammar: https://scholar.google.fr/citations?user=mj06WA0AAAAJ&hl=fr
    
## License
This repo is MIT licensed.
