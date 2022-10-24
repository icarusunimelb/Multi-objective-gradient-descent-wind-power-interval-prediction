import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import seaborn as sns
from sklearn.utils import shuffle
import math
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
import time
from multi_objective_solver import MOSolver
from model import MLP, SNN, qd_objective, LSTM
sns.set(rc = {"figure.figsize" : (24, 18)})

class trainer():
    def __init__(self, modelType='MLP', lambda1_=0.001, lambda2_=0.0008, soften_=160., num_epoch=100, alpha_=0.05, batch_size=128, train_trop=0.8, num_task=2, input_window_size=24, predicted_step=1, num_neurons=64, threshold=0.5, draw=True, display_size=1000):
        self.alpha_ = alpha_
        self.batch_size = batch_size
        self.train_trop = train_trop
        # when num_task == 2, multi objective gradient descent will be applied 
        # when num_task == 1, traditional gradient descent will be applied
        assert num_task in [1, 2]
        self.num_task = num_task
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.modelType = modelType
        if self.modelType == 'MLP':
            self.rnn = False
        else:
            self.rnn = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.soften_ = soften_
        self.num_epoch = num_epoch
        self.draw = draw
        self.display_size = display_size

    def train_test_split(self, country='DE'):
        path = './dataset/' + country + '_supervised_wind_power.csv'
        df = pd.read_csv(path)
        df = df.set_index('index')
        # dataset has already been normalized.

        # split into train/test
        batch_num = df.shape[0]//self.batch_size
        train_batch = int(round(self.train_trop*batch_num))
        test_batch = batch_num - train_batch
        train_size = train_batch * self.batch_size
        test_size = test_batch * self.batch_size
        train = df.head(train_size).values
        test = df.tail(test_size).values

        y_train = train[:,-1].reshape(-1,1)
        y_val = test[:,-1].reshape(-1,1)
        if self.rnn:
            X_train = train[:,:-1].reshape(-1,self.input_window_size,1)
            X_val = test[:,:-1].reshape(-1,self.input_window_size,1)
        else:
            X_train = train[:,:-1]
            X_val = test[:,:-1]
        return X_train, y_train, X_val, y_val
    
    def training_loop(self, X_train, y_train, X_val, y_val, model, optimizer, scheduler, criterion):
        # track the learning rate and train/valid loss
        lrs = []
        train_loss_list_dict = {}
        valid_loss_list_dict = {}
        for i in range(self.num_task):
            train_loss_list_dict[i] = [] 
            valid_loss_list_dict[i] = []

        # training loop    
        for epoch in range(self.num_epoch):
            # training
            train_loss_dict = {}
            for i in range(self.num_task):
                train_loss_dict[i] = 0.0
            model.train()
            x_train, Y_train = shuffle(X_train, y_train)
            for batch in range(math.ceil(X_train.shape[0]/self.batch_size)):
                start = batch * self.batch_size
                end = start + self.batch_size
                inputs = Variable(torch.tensor(x_train[start:end],dtype=torch.float)).to(self.device)
                targets = Variable(torch.tensor(Y_train[start:end],dtype=torch.float)).to(self.device)

                loss_data = {}
                if self.num_task == 2:
                    grads = {}
                    scale = {}
                    
                    for i in range(self.num_task):
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion[i](outputs, targets)
                        loss_data[i] = loss.item()
                        loss.backward()
                        grads[i] = []
                        for param in model.parameters():
                            if param.grad is not None:
                                grads[i].append(Variable(param.grad.data.clone(), requires_grad=False))
                    
                    sol = MOSolver.find_min_norm_element([grads[i] for i in range(self.num_task)])
                    for i in range(self.num_task):
                        scale[i] = float(sol[i])
                        
                    # scaled back-propagation
                    outputs = model(inputs)
                    for i in range(self.num_task):
                        loss_t = criterion[i](outputs, targets)
                        loss_data[i] = loss_t.item()
                        if i > 0:
                            loss = loss + scale[i] * loss_t
                        else: 
                            loss = scale[i] * loss_t
                    loss.backward()
                    optimizer.step()
                else: 
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion[0](outputs, targets)
                    loss_data[0] = loss.item()
                    loss.backward()
                    optimizer.step()

                
                for i in range(self.num_task):
                    train_loss_dict[i] = train_loss_dict[i] + loss_data[i]*inputs.size(0)
            for i in range(self.num_task):
                train_loss_list_dict[i].append(train_loss_dict[i]/X_train.shape[0])
                
            # validation
            valid_loss_dict = {}
            for i in range(self.num_task):
                valid_loss_dict[i] = 0.0
            model.eval()
            for batch in range(math.ceil(X_val.shape[0]/self.batch_size)):
                start = batch * self.batch_size
                end = start + self.batch_size
                inputs = Variable(torch.tensor(X_val[start:end],dtype=torch.float)).to(self.device)
                targets = Variable(torch.tensor(y_val[start:end],dtype=torch.float)).to(self.device)
                outputs = model(inputs)
                for i in range(self.num_task):
                    loss_t = criterion[i](outputs, targets)
                    loss_data[i] = loss_t.item()
                for i in range(self.num_task):
                    valid_loss_dict[i] = valid_loss_dict[i] + loss_data[i]*inputs.size(0)
            for i in range(self.num_task):
                valid_loss_list_dict[i].append(valid_loss_dict[i]/X_val.shape[0])
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            # print logging
            print('------------------------------------------------------------------------------------------------------------------')
            if self.num_task == 2:
                print(f'Epoch {epoch+1} \t\t Weight: {scale[0]} \t\t {scale[1]}')
            for i in range(self.num_task):
                print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss_dict[i] / X_train.shape[0]} \t\t Validation Loss: {valid_loss_dict[i] / X_val.shape[0]}')


        return lrs, train_loss_list_dict, valid_loss_list_dict

    def run(self, country='DE'):
        # create train, test dataset
        X_train, y_train, X_val, y_val = self.train_test_split(country=country)
        # create model
        assert self.modelType in ['MLP', 'SNN', 'LSTM']
        if self.modelType == 'MLP':
            model = MLP(num_neurons = self.num_neurons, input_window_size = self.input_window_size, predicted_step = self.predicted_step)
        elif self.modelType == 'LSTM':
            model = LSTM(num_neurons = self.num_neurons, input_window_size = self.input_window_size, predicted_step = self.predicted_step, device = self.device)
        elif self.modelType == 'SNN':
            model = SNN(num_neurons = self.num_neurons, threshold = self.threshold, input_window_size = self.input_window_size, predicted_step = self.predicted_step)
        # create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.5)
        # create loss function
        criterion = {}
        criterion[0] = qd_objective(lambda_=self.lambda1_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        if self.num_task == 2:
            criterion[1] = qd_objective(lambda_=self.lambda2_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        # to device
        model = model.to(self.device)
        for i in range(self.num_task):
            criterion[i] = criterion[i].to(self.device) 
        # begin training
        lrs, train_loss_list_dict, valid_loss_list_dict = self.training_loop(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, model = model, optimizer = optimizer, scheduler = scheduler, criterion = criterion)

        if self.draw:
            prefix = country + '_' + self.modelType + '_' +str(self.num_task) + 'task_'
            plt.plot(lrs)
            plt.savefig('./fig/'+prefix+'lrs.png')
            plt.show()
            for i in range(self.num_task):
                plt.plot(train_loss_list_dict[i][:], color='r')
                plt.plot(valid_loss_list_dict[i][:], color='b')
                plt.savefig('./fig/'+prefix+'loss'+str(i)+'.png')
                plt.show()
            self.display_size = 1000
            # plot and view some predictions
            y_pred = model(Variable(torch.tensor(X_val[:self.display_size],dtype=torch.float)).to(self.device))
            y_u_pred = y_pred[:,0]
            y_l_pred = y_pred[:,1]

            plt.plot(np.arange(self.display_size),y_val[:self.display_size,0], color='black',label='observations')
            plt.plot(np.arange(self.display_size), y_u_pred.cpu().detach().numpy(), color='b') # upper boundary prediction
            plt.plot(np.arange(self.display_size), y_l_pred.cpu().detach().numpy(), color='b') # lower boundary prediction
            plt.fill_between(np.arange(self.display_size), y_u_pred.cpu().detach().numpy(), y_l_pred.cpu().detach().numpy(), color='b', label='PIs')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("period",fontsize=32)
            plt.ylabel("wind power",fontsize=32)
            plt.legend(loc="upper left",fontsize=32)
            plt.savefig('./fig/'+prefix+'PIs.png')
            plt.show()
        # print some stats
        y_pred = model(Variable(torch.tensor(X_val,dtype=torch.float)).to(self.device))
        y_u_pred = y_pred[:,0]
        y_l_pred = y_pred[:,1]
        K_u = torch.maximum(torch.zeros(1).to(self.device),torch.sign(y_u_pred - Variable(torch.tensor(y_val[:,0],dtype=torch.float)).to(self.device)))
        K_l = torch.maximum(torch.zeros(1).to(self.device),torch.sign(Variable(torch.tensor(y_val[:,0],dtype=torch.float)).to(self.device) - y_l_pred))
        picp = torch.mean(K_u * K_l)
        mpiw = torch.round(torch.mean(torch.absolute(y_u_pred - y_l_pred)),decimals=3)
        print('PICP:', picp)
        print('MPIW:', mpiw)
        return picp, mpiw
        
        
            


        
        


    


