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
Generations produced by two modes of NWGAN:
<p float="left">
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/learnPI_CIFAR_CelebA/samples_mode0_99999.png?raw=true" width="250" />
</figure>
__
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/learnPI_CIFAR_CelebA/samples_mode1_99999.png?raw=true" width="250" />
</figure>
</p>


Generations produced by two modes of baseline model that does not update pi:
<p float="left">
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/fixedPI_CIFAR_CelebA/samples_mode1_99999.png?raw=true" width="250" />
</figure>
__
<figure>
<img src="https://github.com/yogeshbalaji/Normalized-Wasserstein/blob/master/NWGAN/vision/results/fixedPI_CIFAR_CelebA/samples_mode0_99999.png?raw=true" width="250" />
</figure>

</p>


We clearly observe the benefit of updating pi as it makes each generator learn only one mode of the distribution. 
This is useful for several applications e.g., in adversarial clustering. Please refer to the [paper](https://arxiv.org/pdf/1902.00415.pdf) for more details.