# Normalized Wasserstein for domain adaptation
We consider domain adaptation problems under covariate and label shift. 

## Installation
Code requires python3. Please install all packages in requirements.txt by running
 
` pip install -r requirements.txt`

## Data
Download the digits dataset from this [link](http://www.cs.umd.edu/~yogesh/projectpages/normalized_wasserstein/files/digits.zip).
Then, replace the DATA_ROOT variable in scripts/run_mnist_imbalances_*modes.py files
with the path to the downloaded files.

## Training
To run the domain adaptation experiments on 2 GPUs, run

`
    python scripts/run_mnist_imbalanced_3modes.py --ngpus 2
`

ngpus argument specifies the number of GPUs to use. 
Results are sensitive to number of runs. In the given script, we run the experiments 3 times.

## Results
Results are stored in results folder
 
| Method | 3 modes | 5 modes | 10 modes |
| :---: | :---: | :---: | :---: | 
| Source only | 66.63 | 67.44 | 63.17 | 
| DANN | 62.34 | 57.56 | 59.31 |
| Wasserstein | 61.75 | 60.56 | 58.22 |
| **NW** | **75.06** | **76.16** | **68.57** |
