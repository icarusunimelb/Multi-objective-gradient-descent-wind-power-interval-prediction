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
from model import MLP, SNN, qd_objective, LSTM, GRU, winkler_objective, VariationalLSTM, DeepAR, gaussian_log_likelihood
sns.set(rc = {"figure.figsize" : (32, 24)})
plt.rcParams['axes.facecolor'] = 'white'

class trainer():
    def __init__(self, modelType='MLP', trainingType='CrossValidation', lossType='qd', lambda1_=0.001, lambda2_=0.0008, soften_=160., num_epoch=100, alpha_=0.05, fold_size=8, train_prop = 0.8, batch_size=128, num_task=2, input_window_size=24, predicted_step=1, num_neurons=64, threshold=0.5, draw=True, display_size=1000):
        self.alpha_ = alpha_
        if self.alpha_ == 0.05:
            self.n_std_devs = 1.96
        elif self.alpha_ == 0.10:
            self.n_std_devs = 1.645
        elif self.alpha_ == 0.01:
            self.n_std_devs = 2.575
        self.batch_size = batch_size
        # when num_task == 2, multi objective gradient descent will be applied 
        # when num_task == 1, traditional gradient descent will be applied
        assert num_task in [1, 2]
        assert fold_size > 1
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
        self.trainingType = trainingType 
        # cross valiation training or just for a single training
        assert trainingType in ['CrossValidation', 'SinglePass']
        if self.trainingType == 'CrossValidation':
            self.fold_size = fold_size
        elif self.trainingType == 'SinglePass':
            self.fold_size = 1
            self.train_prop = train_prop
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda1_ = lambda1_
        self.lambda2_ = lambda2_
        self.soften_ = soften_
        self.num_epoch = num_epoch
        self.draw = draw
        self.display_size = display_size 
        assert lossType in ['qd', 'winkler','deepAR']
        self.lossType = lossType

    def train_test_split(self, country='DE'):
        path = './dataset/' + str(self.predicted_step)+'_'+country + '_supervised_wind_power.csv'
        df = pd.read_csv(path)
        df = df.set_index('index')
        # dataset has already been normalized.

        X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
        
        if self.trainingType == 'CrossValidation':
            # k-fold cross validation 
            # split into train/test
            batch_num_1fold = (df.shape[0]//(self.batch_size*self.fold_size))
            df = df.head(batch_num_1fold*self.batch_size*self.fold_size)
            for i in range(self.fold_size): 
                train = np.delete(df.values, range(i * batch_num_1fold * self.batch_size, (i+1) * batch_num_1fold * self.batch_size), axis=0)
                test = df.values[i*batch_num_1fold*self.batch_size:(i+1)*batch_num_1fold*self.batch_size]
                print(train.shape, test.shape)

                y_train = train[:,-self.predicted_step:].reshape(-1,self.predicted_step)
                y_val = test[:,-self.predicted_step:].reshape(-1,self.predicted_step)
                if self.rnn:
                    X_train = train[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
                    X_val = test[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
                else:
                    X_train = train[:,:-self.predicted_step]
                    X_val = test[:,:-self.predicted_step]
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                X_val_list.append(X_val)
                y_val_list.append(y_val)
        elif self.trainingType == 'SinglePass':
            batch_num = df.shape[0]//self.batch_size
            train_batch = int(round(self.train_prop*batch_num))
            test_batch = batch_num - train_batch
            train_size = train_batch * self.batch_size
            test_size = test_batch * self.batch_size
            train = df.head(train_size).values
            test = df.tail(test_size).values

            y_train = train[:,-self.predicted_step:].reshape(-1,self.predicted_step)
            y_val = test[:,-self.predicted_step:].reshape(-1,self.predicted_step)
            if self.rnn:
                X_train = train[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
                X_val = test[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
            else:
                X_train = train[:,:-self.predicted_step]
                X_val = test[:,:-self.predicted_step]
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_val_list.append(X_val)
            y_val_list.append(y_val)
        return X_train_list, y_train_list, X_val_list, y_val_list
    
    def training_loop(self, X_train, y_train, X_val, y_val, model, optimizer, scheduler, criterion):
        # track the learning rate and train/valid loss
        lrs = []
        train_loss_list_dict = {}
        valid_loss_list_dict = {}

        train_objective_list_dict = {}
        valid_objective_list_dict = {}
        objective_dict = {}
        for i in range(self.num_task):
            train_loss_list_dict[i] = [] 
            valid_loss_list_dict[i] = []
        for i in range(2):
            train_objective_list_dict[i] = []
            valid_objective_list_dict[i] = []
            objective_dict[i] = 0.0

        # training loop    
        for epoch in range(self.num_epoch):
            # training
            train_loss_dict = {}
            for i in range(self.num_task):
                train_loss_dict[i] = 0.0
            model.train()
            x_train, Y_train = shuffle(X_train, y_train)
            for i in range(2):
                objective_dict[i] = 0.0
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
                elif self.lossType == 'deepAR' : 
                    optimizer.zero_grad()
                    mu, sigma = model(inputs)
                    outputs = torch.cat([mu+self.n_std_devs*sigma, mu-self.n_std_devs*sigma],1)
                    loss = criterion[0](mu, sigma, targets)
                    loss_data[0] = loss.item()
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
                K_u = torch.maximum(torch.zeros(1).to(self.device),torch.sign(outputs[:,0] - targets[:,0]))
                K_l = torch.maximum(torch.zeros(1).to(self.device),torch.sign(targets[:,0] - outputs[:,1]))
                if self.lossType == "winkler":
                    # winkler loss
                    objective_dict[0] =  objective_dict[0] + torch.sum(torch.abs(outputs[:,0]-outputs[:,1]) + (2/self.alpha_)*(torch.multiply(outputs[:,1]-targets[:,0], torch.maximum(torch.zeros(1).to(self.device),torch.sign(outputs[:,1] - targets[:,0])))) + (2/self.alpha_)*(torch.multiply(targets[:,0]-outputs[:,0], torch.maximum(torch.zeros(1).to(self.device),torch.sign(targets[:,0] - outputs[:,0])))))
                    picp = torch.mean(K_u * K_l)
                    # coverage probability constraint
                    objective_dict[1] = objective_dict[1] + self.batch_size*self.batch_size/ (self.alpha_ * (1-self.alpha_)) * torch.square((1-self.alpha_) - picp)
                else:
                    # PICP
                    objective_dict[0] = objective_dict[0] + torch.mean(torch.multiply(K_u, K_l))*self.batch_size
                    # MPIW
                    objective_dict[1] = objective_dict[1] + torch.sum(torch.abs(outputs[:,0] - outputs[:,1]))
            for i in range(self.num_task):
                train_loss_list_dict[i].append(train_loss_dict[i]/X_train.shape[0])
            for i in range(2):   
                train_objective_list_dict[i].append((objective_dict[i]/X_train.shape[0]).cpu().detach())
                
            # validation
            valid_loss_dict = {}
            for i in range(self.num_task):
                valid_loss_dict[i] = 0.0
            model.eval()
            for i in range(2):
                objective_dict[i] = 0.0
            for batch in range(math.ceil(X_val.shape[0]/self.batch_size)):
                start = batch * self.batch_size
                end = start + self.batch_size
                inputs = Variable(torch.tensor(X_val[start:end],dtype=torch.float)).to(self.device)
                targets = Variable(torch.tensor(y_val[start:end],dtype=torch.float)).to(self.device)
                if self.lossType == 'deepAR':
                    mu, sigma = model(inputs)
                    outputs = torch.cat([mu+self.n_std_devs*sigma, mu-self.n_std_devs*sigma],1)
                else:
                    outputs = model(inputs)
                for i in range(self.num_task):
                    if self.lossType == 'deepAR':
                        loss_t = criterion[i](mu, sigma, targets)
                    else:
                        loss_t = criterion[i](outputs, targets)
                    loss_data[i] = loss_t.item()
                for i in range(self.num_task):
                    valid_loss_dict[i] = valid_loss_dict[i] + loss_data[i]*inputs.size(0)
                K_u = torch.maximum(torch.zeros(1).to(self.device),torch.sign(outputs[:,0] - targets[:,0]))
                K_l = torch.maximum(torch.zeros(1).to(self.device),torch.sign(targets[:,0] - outputs[:,1]))
                if self.lossType == "winkler":
                    # winkler loss
                    objective_dict[0] =  objective_dict[0] + torch.sum(torch.abs(outputs[:,0]-outputs[:,1]) + (2/self.alpha_)*(torch.multiply(outputs[:,1]-targets[:,0], torch.maximum(torch.zeros(1).to(self.device),torch.sign(outputs[:,1] - targets[:,0])))) + (2/self.alpha_)*(torch.multiply(targets[:,0]-outputs[:,0], torch.maximum(torch.zeros(1).to(self.device),torch.sign(targets[:,0] - outputs[:,0])))))
                    picp = torch.mean(K_u * K_l)
                    # coverage probability constraint
                    objective_dict[1] = objective_dict[1] + self.batch_size*self.batch_size/ (self.alpha_ * (1-self.alpha_)) * torch.square((1-self.alpha_) - picp)
                else: 
                    # DeepAR and Coverage-width critrion methods all use this for validation. 
                    # PICP
                    objective_dict[0] = objective_dict[0] + torch.mean(torch.multiply(K_u, K_l))*self.batch_size
                    # MPIW
                    objective_dict[1] = objective_dict[1] + torch.sum(torch.abs(outputs[:,0] - outputs[:,1]))
            for i in range(self.num_task):
                valid_loss_list_dict[i].append(valid_loss_dict[i]/X_val.shape[0])
            for i in range(2):   
                valid_objective_list_dict[i].append((objective_dict[i]/X_val.shape[0]).cpu().detach())
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            # print logging
            print('------------------------------------------------------------------------------------------------------------------')
            if self.num_task == 2:
                print(f'Epoch {epoch+1} \t\t Weight: {scale[0]} \t\t {scale[1]}')
            for i in range(self.num_task):
                print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss_dict[i] / X_train.shape[0]} \t\t Validation Loss: {valid_loss_dict[i] / X_val.shape[0]}')


        return lrs, train_loss_list_dict, valid_loss_list_dict, train_objective_list_dict, valid_objective_list_dict
    
    def one_fold_training(self, X_train, y_train, X_val, y_val, country='DE'):
        # create model
        assert self.modelType in ['MLP', 'SNN', 'LSTM', 'BiGRU']
        if self.modelType == 'MLP':
            model = MLP(num_neurons = self.num_neurons, input_window_size = self.input_window_size, predicted_step = self.predicted_step)
        elif self.modelType == 'LSTM':
            model = LSTM(num_neurons = self.num_neurons, input_window_size = self.input_window_size, predicted_step = self.predicted_step, device = self.device)
        elif self.modelType == 'SNN':
            model = SNN(num_neurons = self.num_neurons, threshold = self.threshold, input_window_size = self.input_window_size, predicted_step = self.predicted_step)
        elif self.lossType == 'deepAR':
            # DeepAR in our work also employs BiGRU, which is different from its original work.
            model = DeepAR(num_neurons = self.num_neurons,input_window_size = self.input_window_size, predicted_step = self.predicted_step,bidirectional = True, device = self.device) 
        elif self.modelType == 'BiGRU':
            model = GRU(num_neurons = self.num_neurons, input_window_size = self.input_window_size, predicted_step = self.predicted_step, bidirectional = True, device = self.device)
        # create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # employ learning rate scheduler, the default scheduler wonnt change the learning rate, feel free to modify it
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
        # create loss function
        criterion = {}
        assert self.lossType in ['qd','winkler', 'deepAR']
        if self.lossType == 'qd':
            criterion[0] = qd_objective(lambda_=self.lambda1_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
            if self.num_task == 2:
                criterion[1] = qd_objective(lambda_=self.lambda2_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        elif self.lossType == 'winkler':
            criterion[0] = winkler_objective(lambda_=self.lambda1_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
            if self.num_task == 2:
                criterion[1] = winkler_objective(lambda_=self.lambda2_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        elif self.lossType == 'deepAR':
            # DeepAR in our work will not employ multi objective gradient descent. 
            criterion[0] = gaussian_log_likelihood() 

        # to device
        model = model.to(self.device)
        for i in range(self.num_task):
            criterion[i] = criterion[i].to(self.device) 
        
        # begin training
        lrs, train_loss_list_dict, valid_loss_list_dict, train_objective_list_dict, valid_objective_list_dict= self.training_loop(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, model = model, optimizer = optimizer, scheduler = scheduler, criterion = criterion)

        if self.draw and self.predicted_step==1:
            
            '''
 
            # tracking two intermediate loss functions
            for i in range(self.num_task):
                plt.plot(train_loss_list_dict[i][:], color='r', linewidth=3, label='Training loss')
                plt.plot(valid_loss_list_dict[i][:], color='b', linewidth=3, label='Validation loss')
                plt.xlabel("Training epoches",fontsize=64)
                plt.ylabel("Training loss",fontsize=64)
                plt.xticks(fontsize = 42)
                plt.yticks(fontsize = 42)
                plt.legend(loc="upper left",fontsize=64)
                plt.show()
            '''
            '''
            # tracking winkler loss and cpc objectives 
            plt.plot(train_objective_list_dict[0][:], color='r', linewidth=3, label='Training loss')
            plt.plot(valid_objective_list_dict[0][:], color='b', linewidth=3, label='Validation loss')
            plt.xlabel("Training epoches",fontsize=64)
            plt.ylabel("Winkler loss",fontsize=64)
            plt.xticks(fontsize = 42)
            plt.yticks(fontsize = 42)
            plt.legend(loc="upper left",fontsize=64)
            plt.show()

            plt.plot(train_objective_list_dict[1][:], color='r', linewidth=3, label='Training loss')
            plt.plot(valid_objective_list_dict[1][:], color='b', linewidth=3, label='Validation loss')
            plt.xlabel("Training epoches",fontsize=64)
            plt.ylabel("Coverage probability constraint",fontsize=64)
            plt.xticks(fontsize = 42)
            plt.yticks(fontsize = 42)
            plt.legend(loc="upper left",fontsize=64)
            plt.show()
            '''

            self.display_size = 200
            # plot and view some predictions
            if self.lossType == 'deepAR':
                mu, sigma = model(Variable(torch.tensor(X_val[300:300+self.display_size],dtype=torch.float)).to(self.device))
                y_u_pred = mu+self.n_std_devs*sigma
                y_l_pred = mu-self.n_std_devs*sigma
            else:
                y_pred = model(Variable(torch.tensor(X_val[300:300+self.display_size],dtype=torch.float)).to(self.device))
                y_u_pred = y_pred[:,::2]
                y_l_pred = y_pred[:,1::2]

            plt.plot(np.arange(self.display_size),y_val[300:300+self.display_size,0],linewidth=3, color='black',label='Observations')

            plt.plot(np.arange(self.display_size), y_u_pred.cpu().detach().numpy().flatten(), color='#33FFE3') # upper boundary prediction
            plt.plot(np.arange(self.display_size), y_l_pred.cpu().detach().numpy().flatten(), color='#33FFE3') # lower boundary prediction
            plt.fill_between(np.arange(self.display_size), y_u_pred.cpu().detach().numpy().flatten(), y_l_pred.cpu().detach().numpy().flatten(), color='#33FFE3', label='Prediction intervals')
            plt.xlabel("Time(hour)",fontsize=64)
            plt.ylabel("Normalized wind power",fontsize=64)
            plt.xticks(fontsize = 42)
            plt.yticks(fontsize = 42)
            plt.legend(loc="upper right",fontsize=64)
            plt.show()
        elif self.draw and self.predicted_step > 1: 
            self.display_size = 1
            N = 5
            # plot and view some predictions
            if self.lossType == 'deepAR':
                mu, sigma = model(Variable(torch.tensor(X_val[300:300+self.display_size],dtype=torch.float)).to(self.device))
                y_u_pred = mu+self.n_std_devs*sigma
                y_l_pred = mu-self.n_std_devs*sigma
            else:
                y_pred = model(Variable(torch.tensor(X_val[300:300+self.display_size],dtype=torch.float)).to(self.device))
                y_u_pred = y_pred[:,::2]
                y_l_pred = y_pred[:,1::2]
            
            time_series = y_val[300,:]
            for l in range(N):
                time_series = np.concatenate((X_val[300-l*self.input_window_size,:,0],time_series))
                
            plt.plot(np.arange(N*self.input_window_size+self.predicted_step),time_series,linewidth=3, color='black',label='Observations')

            plt.plot(np.arange((N-1)*self.input_window_size+self.input_window_size,N*self.input_window_size+self.predicted_step), y_u_pred.cpu().detach().numpy().flatten(), color='#33FFE3') # upper boundary prediction
            plt.plot(np.arange((N-1)*self.input_window_size+self.input_window_size,N*self.input_window_size+self.predicted_step), y_l_pred.cpu().detach().numpy().flatten(), color='#33FFE3') # lower boundary prediction
            plt.fill_between(np.arange((N-1)*self.input_window_size+self.input_window_size,N*self.input_window_size+self.predicted_step), y_u_pred.cpu().detach().numpy().flatten(), y_l_pred.cpu().detach().numpy().flatten(), color='#33FFE3', label='Prediction intervals')
            plt.xlabel("Time(hour)",fontsize=64)
            plt.ylabel("Normalized wind power",fontsize=64)
            plt.xticks(fontsize = 42)
            plt.yticks(fontsize = 42)
            plt.legend(loc="upper right",fontsize=64)
            plt.show()
    
        # print some stats
        if self.lossType == 'deepAR':
            mu, sigma = model(Variable(torch.tensor(X_val,dtype=torch.float)).to(self.device))
            y_u_pred = mu+self.n_std_devs*sigma
            y_l_pred = mu-self.n_std_devs*sigma
        else:
            y_pred = model(Variable(torch.tensor(X_val,dtype=torch.float)).to(self.device))
            y_u_pred = y_pred[:,::2]
            y_l_pred = y_pred[:,1::2]
        K_u = torch.maximum(torch.zeros(1).to(self.device),torch.sign(y_u_pred - Variable(torch.tensor(y_val,dtype=torch.float)).to(self.device)))
        K_l = torch.maximum(torch.zeros(1).to(self.device),torch.sign(Variable(torch.tensor(y_val,dtype=torch.float)).to(self.device) - y_l_pred))
        picp = torch.mean(K_u * K_l)
        mpiw = torch.round(torch.mean(torch.absolute(y_u_pred - y_l_pred)),decimals=3)
        S_t = torch.abs(y_u_pred-y_l_pred) + (2/self.alpha_)*(torch.multiply(y_l_pred-Variable(torch.tensor(y_val,dtype=torch.float)).to(self.device), torch.maximum(torch.zeros(1).to(self.device),torch.sign(y_l_pred - Variable(torch.tensor(y_val,dtype=torch.float)).to(self.device))))) + (2/self.alpha_)*(torch.multiply(Variable(torch.tensor(y_val,dtype=torch.float)).to(self.device)-y_u_pred, torch.maximum(torch.zeros(1).to(self.device),torch.sign(Variable(torch.tensor(y_val,dtype=torch.float)).to(self.device) - y_u_pred))))
        S_overline = torch.mean(S_t)
        print('PICP:', picp)
        print('MPIW:', mpiw)
        print('Winkler Score:',S_overline)
        return picp.cpu().detach().numpy(), mpiw.cpu().detach().numpy(), S_overline.cpu().detach().numpy(), train_objective_list_dict, valid_objective_list_dict

    def run(self, country='DE'):
        # create train, test dataset
        X_train_list, y_train_list, X_val_list, y_val_list = self.train_test_split(country=country)
        picp_list = []
        mpiw_list = []
        ace_list = []
        score_list = []
        for i in range(self.fold_size):
            print('--------------------------------------------------fold'+str(i)+'---------------------------------------------------------')
            picp, mpiw, score = self.one_fold_training(X_train_list[i], y_train_list[i], X_val_list[i], y_val_list[i], country = country)
            picp_list.append(picp)
            mpiw_list.append(mpiw)  
            ace_list.append(picp-(1-self.alpha_))
            score_list.append(score)  
        print("picp mean: "+str(np.mean(picp_list)))
        print("picp std: "+str(np.std(picp_list)))
        print("mpiw mean: "+str(np.mean(mpiw_list)))
        print("mpiw std: "+str(np.std(mpiw_list)))
        print("ace mean: "+str(np.mean(ace_list)))
        print("ace std: "+str(np.std(ace_list)))
        print("winkler score mean: "+str(np.mean(score_list)))
        print("winkler score std: "+str(np.std(score_list)))
        return np.mean(picp_list), np.std(picp_list), np.mean(mpiw_list), np.std(mpiw_list), np.mean(ace_list), np.std(ace_list), np.mean(score_list), np.std(score_list)
        
            
        

        
        
            


        
        


    


