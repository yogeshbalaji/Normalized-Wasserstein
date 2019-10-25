# Normalized Wasserstein for generative models
In this experiment, we train NWGAN on a mixture of CIFAR-10 and CelebA datasets

## Installation
Code requires **python2**. Please install all packages in requirements.txt by running
 
` pip install -r requirements.txt`

## Training
To training NWGAN on mixture of CIFAR-10 and CelebA, run

`
    python main.py --cfg-path configs/CIFAR_CelabA_learnPI.json
`

As a baseline, we compare it with a model not updating the mixture proportion, but fixing it to [0.5, 0.5]
To run, this baseline, run

`
    python main.py --cfg-path configs/CIFAR_CelabA_fixedPI.json
`

## Results
Generations produced by NWGAN:
