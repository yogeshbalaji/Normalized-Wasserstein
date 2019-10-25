# Normalized Wasserstein for generative models
In this experiment, we train NWGAN on a mixture of CIFAR-10 and CelebA datasets

## Installation
Code requires **python2.7**. Please install all packages in requirements.txt by running
 
` pip install -r requirements.txt`

## Training
To training NWGAN on mixture of CIFAR-10 and CelebA, run

`
    python main.py --cfg-path configs/CIFAR_CelebA_learnPI.json
`

As a baseline, we compare it with a model not updating the mixture proportion, but fixing it to [0.5, 0.5]
To run, this baseline, run

`
    python main.py --cfg-path configs/CIFAR_CelabA_fixedPI.json
`

## Results
Generations produced by two modes of NWGAN:
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/learnPI_CIFAR_CelebA/samples_mode0_99999.png?raw=true" width="250">
</figure>
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/learnPI_CIFAR_CelebA/samples_mode1_99999.png?raw=true" width="250">
</figure>



Generations produced by two modes of baseline model that assumes uniform mode proportion (pi):
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/fixedPI_CIFAR_CelebA/samples_mode1_99999.png?raw=true" width="250">
</figure>
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/fixedPI_CIFAR_CelebA/samples_mode0_99999.png?raw=true" width="250">
</figure>


We clearly observe the benefit of updating mode proprotion(pi) in optimization as it makes each generator learn only one mode of the distribution. 
This is useful for several applications e.g., in adversarial clustering. Please refer to the [paper](https://arxiv.org/pdf/1902.00415.pdf) for more details.
