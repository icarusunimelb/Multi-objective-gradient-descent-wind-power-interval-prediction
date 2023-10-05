import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate 

class winkler_objective(nn.Module):
    "Constrainted Winkler loss function"
    def __init__(self, lambda_ = 0.001, alpha_ = 0.05, soften_=160., device='cpu', batch_size=128):
        super(winkler_objective, self).__init__()
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.soften_ = soften_
        self.device = device
        self.batch_size = batch_size
    
    def forward(self, y_pred, y_true):
        y_true = y_true[:,0]
        y_u = y_pred[:,0]
        y_l = y_pred[:,1]

        K_SU = torch.sigmoid(self.soften_ * (y_u - y_true))
        K_SL = torch.sigmoid(self.soften_ * (y_true - y_l))
        K_S = torch.multiply(K_SU, K_SL)
        
        PICP_S = torch.mean(K_S)
        MLE_PICP = self.batch_size / (self.alpha_ * (1-self.alpha_)) * torch.square((1-self.alpha_) - PICP_S)

        S_t = torch.abs(y_u-y_l) + (2/self.alpha_)*(torch.multiply(y_l-y_true, torch.sigmoid(self.soften_ * (y_l - y_true)))) + (2/self.alpha_)*(torch.multiply(y_true-y_u, torch.sigmoid(self.soften_ * (y_true - y_u))))
        S_overline = torch.mean(S_t)

        Loss = S_overline + self.lambda_ * MLE_PICP 

        return Loss



class qd_objective(nn.Module):
    '''Loss_QD'''
    def __init__(self, lambda_ = 0.001, alpha_ = 0.05, soften_=160., device='cpu', batch_size=128):
        super(qd_objective, self).__init__()
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.soften_ = soften_
        self.epsilon =  torch.finfo(torch.float).eps
        self.device = device
        self.batch_size = batch_size
    
    def forward(self, y_pred, y_true):
        y_true = y_true[:,0]
        y_u = y_pred[:,0]
        y_l = y_pred[:,1]

        K_HU = torch.maximum(torch.zeros(1).to(self.device),torch.sign(y_u - y_true))
        K_HL = torch.maximum(torch.zeros(1).to(self.device),torch.sign(y_true - y_l))
        K_H = torch.multiply(K_HU, K_HL)

        K_SU = torch.sigmoid(self.soften_ * (y_u - y_true))
        K_SL = torch.sigmoid(self.soften_ * (y_true - y_l))
        K_S = torch.multiply(K_SU, K_SL)
        
        PICP_S = torch.mean(K_S)

        MPIW_c = torch.sum(torch.multiply((y_u - y_l),K_H))/(torch.sum(K_H)+self.epsilon)
        MLE_PICP = self.batch_size / (self.alpha_ * (1-self.alpha_)) * torch.square(torch.maximum(torch.zeros(1).to(self.device),(1-self.alpha_) - PICP_S))
        
        Loss_S = MPIW_c + self.lambda_ * MLE_PICP
        
        return Loss_S

class MLP(nn.Module):

    def __init__(self, num_neurons = 64, input_window_size = 24, predicted_step = 1):
        super(MLP, self).__init__()
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        # an affine operation: y = Wx + b
        self.num_neurons = num_neurons
        self.fc1 = nn.Linear(self.input_window_size, self.num_neurons)
        self.bn1 = nn.BatchNorm1d(self.num_neurons)
        self.fc2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.bn2= nn.BatchNorm1d(self.num_neurons)
        self.fc3 = nn.Linear(self.num_neurons, self.num_neurons)
        self.bn3= nn.BatchNorm1d(self.num_neurons)
        self.output = nn.Linear(self.num_neurons, 2*self.predicted_step)
        self.output.bias = torch.nn.Parameter(torch.tensor([0.2,-0.2]))

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.output(x)
        return x

# https://arxiv.org/pdf/1512.05287.pdf
# To adapt from NLP task to univariable time series forecasting task, the embedding dropout is removed and the NN architecture is adapted to keep simplicity.   
class VariationalDropout(nn.Module):
    """
    Variational Dropout module. In comparison to the default PyTorch module, this one only changes the dropout mask when
    sample() is called.
    """

    def __init__(self, dropout, input_dim, device):
        super().__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.device = device
        self.mask = None

    def forward(self, x):
        if self.mask is None:
            raise ValueError("Dropout mask hasn't been sampled yet. Use .sample().")

        return (x * self.mask)

    def sample(self, batch_size: int):
        """
        Sample a new dropout mask for a batch of specified size.
        Parameters
        ----------
        batch_size: int
            Size of current batch.
        """
        self.mask = (torch.bernoulli(
            torch.ones(batch_size, self.input_dim, device=self.device)
            * (1 - self.dropout)
        ) / (1 - self.dropout))

class VariationalLSTM(nn.Module):

    def __init__(self, num_neurons = 64, input_window_size = 24, predicted_step = 1, layer_dropout = 0.2, time_dropout = 0.2, batch_size=128, device = 'cpu'):
        super(VariationalLSTM, self).__init__()
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.num_neurons = num_neurons
        self.device = device
        self.layer_dropout = layer_dropout
        self.time_dropout = time_dropout
        self.batch_size = batch_size
        self.lstm1 = nn.LSTMCell(1, self.num_neurons)
        self.lstm2 = nn.LSTMCell(self.num_neurons, self.num_neurons)
        self.output = nn.Linear(self.num_neurons, 2*self.predicted_step)
        self.output.bias = torch.nn.Parameter(torch.tensor([0.2,-0.2]))
        # dropout modules 
        num_layers = 2
        self.dropout_modules = {
            "layer": [VariationalDropout(layer_dropout, num_neurons, device) for _ in range(num_layers)],
            "time": [VariationalDropout(time_dropout, num_neurons, device) for _ in range(num_layers)],
        } 

    def forward(self, x):   
        # batch_size x hidden_size
        hidden_state_1 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        cell_state_1 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        hidden_state_2 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        cell_state_2 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        
        # weights initialization
        torch.nn.init.xavier_normal_(hidden_state_1)
        torch.nn.init.xavier_normal_(cell_state_1)
        torch.nn.init.xavier_normal_(hidden_state_2)
        torch.nn.init.xavier_normal_(cell_state_2)

        # sample dropout masks
        self.sample_masks(self.batch_size)
        
        # unfolding LSTM
        for i in range(self.input_window_size):
            hidden_state_1, cell_state_1 = self.lstm1(x[:, i], (self.dropout_modules["time"][0](hidden_state_1), cell_state_1))
            hidden_state_1 = self.dropout_modules["layer"][0](hidden_state_1)
            hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (self.dropout_modules["time"][1](hidden_state_2), cell_state_2))
            hidden_state_2 = self.dropout_modules["layer"][1](hidden_state_2)
        output = self.output(hidden_state_2)
        return output
    
    def sample_masks(self, batch_size):
        """
        Sample masks for the current batch.
        Parameters
        ----------
        batch_size: int
            Size of the current batch.
        """
        # Iterate over type of dropout modules ("layer", "time")
        for dropout_modules in self.dropout_modules.values():
            # Iterate over all dropout modules of one type (across different layers)
            for layer_module in dropout_modules:
                layer_module.sample(batch_size)

class LSTM(nn.Module):

    def __init__(self, num_neurons = 64, input_window_size = 24, predicted_step = 1, device = 'cpu'):
        super(LSTM, self).__init__()
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.num_neurons = num_neurons
        self.device = device
        self.lstm1 = nn.LSTMCell(1, self.num_neurons)
        self.lstm2 = nn.LSTMCell(self.num_neurons, self.num_neurons)
        self.output = nn.Linear(self.num_neurons, 2*self.predicted_step)
        self.output.bias = torch.nn.Parameter(torch.tensor([0.2,-0.2]))

    def forward(self, x):   
        # batch_size x hidden_size
        hidden_state_1 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        cell_state_1 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        hidden_state_2 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        cell_state_2 = torch.zeros(x.size(0), self.num_neurons).to(self.device)
        
        # weights initialization
        torch.nn.init.xavier_normal_(hidden_state_1)
        torch.nn.init.xavier_normal_(cell_state_1)
        torch.nn.init.xavier_normal_(hidden_state_2)
        torch.nn.init.xavier_normal_(cell_state_2)
        
        # unfolding LSTM
        for i in range(self.input_window_size):
            hidden_state_1, cell_state_1 = self.lstm1(x[:, i], (hidden_state_1, cell_state_1))
            hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (hidden_state_2, cell_state_2))
        output = self.output(hidden_state_2)
        return output

class GRU(nn.Module):
    # Note the GRU used in the paper is Bidirectional GRU
    def __init__(self, num_neurons = 64, input_window_size = 24, predicted_step = 1, layer_num = 2, bidirectional = True, device = 'cpu'):
        super(GRU, self).__init__()
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.num_neurons = num_neurons
        self.device = device
        self.layer_num = layer_num
        self.D = 1
        if bidirectional:
            self.D = 2
        self.gru = nn.GRU(1, self.num_neurons, self.layer_num, batch_first=True, bidirectional = bidirectional)
        self.output = nn.Linear(self.D*self.num_neurons, 2*self.predicted_step)
        # When meet init issue in qd objective, can uncomment the following code 
        self.output.bias = torch.nn.Parameter(torch.tensor([0.2,-0.2]))

    def forward(self, x):  
        # Initializing hidden state for first input with zeros
        hidden_state0 = torch.zeros(self.D*self.layer_num, x.size(0), self.num_neurons).to(self.device)
        output, _ = self.gru(x, hidden_state0)
        output = self.output(output[:,-1,:])
        return output

class SNN(nn.Module):

    def __init__(self, num_neurons = 64, threshold = 0.5, input_window_size = 24, predicted_step = 1):
        super(SNN, self).__init__()
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.slstm1 = snn.SLSTM(1, self.num_neurons, threshold=self.threshold, spike_grad=surrogate.fast_sigmoid(), learn_threshold=True)
        self.slstm2 = snn.SLSTM(self.num_neurons, self.num_neurons, threshold=self.threshold, spike_grad=surrogate.fast_sigmoid(), learn_threshold=True)
        self.output = nn.Linear((self.input_window_size+2)*self.num_neurons, 2*self.predicted_step)
        self.output.bias = torch.nn.Parameter(torch.tensor([0.2,-0.2]))

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        
        lst = None
        
        for step in range(self.input_window_size):
            spk1, syn1, mem1 = self.slstm1(x[:, step, :], syn1, mem1)
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)
           
            if lst == None:
                lst = spk2
            else:
                lst = torch.cat((lst, spk2), dim=1)
                
        spk2, syn2, mem2 = self.slstm2(mem1, syn2, mem2)
        lst = torch.cat((lst, spk2), dim=1)        
        lst = torch.cat((lst, mem2), dim=1)
        
        otp = self.output(lst)
        return otp
