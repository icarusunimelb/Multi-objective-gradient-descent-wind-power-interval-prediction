# Multi-objective-gradient-descent-wind-power-interval-prediction
This repository contains the source code for multi-objective gradient descent-based lower upper bound estimation (MOGD-LUBE) approach for wind power interval prediction. The original paper can be found using following link: .
A script [run.ipynb](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/run.ipynb) is provided to test the code. To apply the model on other datasets, please refer to the [preprocessing.py](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/preprocessing.py) to generate time series dataset with suitable format and modify the dataset path in [trainer.py](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/trainer.py).  

## Dependencies
MOGD-LUBE leans heavily on the following Python packages: python3, numpy, pandas, scikit-learn, matplotlib, math, pytorch, torchvision

To use the spiking neural network model, please install the [snntorch](https://snntorch.readthedocs.io/en/latest). 

## Reference 
If you use MOGD-LUBE for your research, I would appreciate it if you would cite the following paper: 

using the following BibTex: 

## Licence
MOGD-LUBE is licensed under the open source [MIT License](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/LICENSE.md).
