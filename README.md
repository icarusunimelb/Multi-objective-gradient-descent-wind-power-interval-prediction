# Multi-objective-gradient-descent-wind-power-interval-prediction
This repository contains the source code for multi-objective gradient descent-based lower upper bound estimation (MOGD-LUBE) approach for wind power interval prediction. The original paper can be found using following link: .
A script [run.ipynb](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/run.ipynb) is provided to test the code. To apply the model on other datasets, please refer to the [preprocessing.py](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/preprocessing.py) to generate time series dataset with suitable format and modify the dataset path in [trainer.py](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/trainer.py).  

## Dependencies
MOGD-LUBE leans heavily on the following Python packages: python3, numpy, pandas, scikit-learn, matplotlib, math, pytorch, torchvision

To use the spiking neural network model, please install the [snntorch](https://snntorch.readthedocs.io/en/latest). 

## Reference 
If you use MOGD-LUBE for your research, I would appreciate it if you would cite the following paper: 

using the following BibTex:
```
    @ARTICLE{10268627,
      author={Chen, Yinsong and Yu, Samson S. and Lim, Chee Peng and Shi, Peng},
      journal={IEEE Transactions on Sustainable Energy}, 
      title={Multi-Objective Estimation of Optimal Prediction Intervals for Wind Power Forecasting}, 
      year={2023},
      volume={},
      number={},
      pages={1-12},
      doi={10.1109/TSTE.2023.3321081}}
```
and 
```
    @INPROCEEDINGS{10181537,
      author={Chen, Yinsong and Yu, Samson and Eshraghian, Jason K. and Lim, Chee Peng},
      booktitle={2023 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
      title={Multi-Objective Spiking Neural Network for Optimal Wind Power Prediction Interval}, 
      year={2023},
      volume={},
      number={},
      pages={1-5},
      doi={10.1109/ISCAS46773.2023.10181537}}
```

## Licence
MOGD-LUBE is licensed under the open source [MIT License](https://github.com/icarusunimelb/Multi-objective-gradient-descent-wind-power-interval-prediction/blob/main/LICENSE.md).
