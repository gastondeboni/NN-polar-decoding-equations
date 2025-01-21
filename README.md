# NN-polar-decoding-equations
Implementation of neural networks for estimating the decoding equations of polar codes with arbitrary kernel size.

## Disclaimer
1) The repository is in constant change. The uploaded model was trained using TensorFlow 2.18.0.
2) This repository is only intended to train the neural networks that approximate the marginalization functions. It does not constitute a complete Polar decoder but only a building block.

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

Alternatively, you can use the models already in the repo to build your decoder for polar code generated using a kernel of size 16.

## Reference
The paper is currently under review for ISIT 2025.

## Authors info
Valerio Bioglio: https://scholar.google.com/citations?user=ZMGylHoAAAAJ&hl=en <br />
Gastón De Boni Rovella: https://scholar.google.com/citations?user=ZcM9QRUAAAAJ&hl=en&authuser=1 <br />
Meryem Benammar: https://scholar.google.fr/citations?user=mj06WA0AAAAJ&hl=fr
    
## License
This repo is MIT licensed.
