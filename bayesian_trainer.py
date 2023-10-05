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
import time
from model import qd_objective, winkler_objective, VariationalLSTM
sns.set(rc = {"figure.figsize" : (32, 24)})
plt.rcParams['axes.facecolor'] = 'white'

class bayesian_trainer():
    # batch_size == display_size to avoid extra batchize during visualization
    def __init__(self, modelType='VariationalLSTM', trainingType='CrossValidation', lossType='winkler', layer_dropout = 0.2, time_dropout = 0.2, num_forward_passes = 100, lambda_ = 0.0005, soften_=160., num_epoch=100, alpha_=0.05, fold_size=8, train_prop = 0.8, batch_size=128, input_window_size=24, predicted_step=1, num_neurons=64, draw=True, display_size=128):
        self.alpha_ = alpha_
        self.batch_size = batch_size
        assert fold_size > 1
        self.input_window_size = input_window_size
        self.predicted_step = predicted_step
        self.num_neurons = num_neurons
        self.modelType = modelType
        self.trainingType = trainingType 
        # cross valiation training or just for a single training
        assert trainingType in ['CrossValidation', 'SinglePass']
        if self.trainingType == 'CrossValidation':
            self.fold_size = fold_size
        elif self.trainingType == 'SinglePass':
            self.fold_size = 1
            self.train_prop = train_prop
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_ = lambda_
        self.soften_ = soften_
        self.num_epoch = num_epoch
        self.draw = draw
        self.display_size = display_size 
        assert lossType in ['qd', 'winkler']
        self.lossType = lossType
        self.layer_dropout = layer_dropout
        self.time_dropout = time_dropout
        self.num_forward_passes = num_forward_passes

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
                    X_train = train[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
                    X_val = test[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
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
                X_train = train[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
                X_val = test[:,:-self.predicted_step].reshape(-1,self.input_window_size,1)
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                X_val_list.append(X_val)
                y_val_list.append(y_val)
            return X_train_list, y_train_list, X_val_list, y_val_list

    def training_loop(self, X_train, y_train, X_val, y_val, model, optimizer, scheduler, criterion):
        # track the learning rate and train/valid loss
        lrs = []
        train_loss_list = []
        valid_loss_list = []

        # training loop    
        for epoch in range(self.num_epoch):
            # training
            train_loss = 0.0
            model.train()
            x_train, Y_train = shuffle(X_train, y_train)
            for batch in range(math.ceil(X_train.shape[0]/self.batch_size)):
                start = batch * self.batch_size
                end = start + self.batch_size
                inputs = Variable(torch.tensor(x_train[start:end],dtype=torch.float)).to(self.device)
                targets = Variable(torch.tensor(Y_train[start:end],dtype=torch.float)).to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_data = loss.item()
                loss.backward()
                optimizer.step()
 
                train_loss = train_loss + loss_data*inputs.size(0)
            train_loss_list.append(train_loss/X_train.shape[0])
                
            # validation
            valid_loss = 0.0
            model.eval()
            for batch in range(math.ceil(X_val.shape[0]/self.batch_size)):
                start = batch * self.batch_size
                end = start + self.batch_size
                inputs = Variable(torch.tensor(X_val[start:end],dtype=torch.float)).to(self.device)
                targets = Variable(torch.tensor(y_val[start:end],dtype=torch.float)).to(self.device)
                outputs = model(inputs)
                loss_t = criterion(outputs, targets)
                loss_data = loss_t.item()
                valid_loss = valid_loss + loss_data*inputs.size(0)
            valid_loss_list.append(valid_loss/X_val.shape[0])
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            # print logging
            print('------------------------------------------------------------------------------------------------------------------')
            print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / X_train.shape[0]} \t\t Validation Loss: {valid_loss / X_val.shape[0]}')

        return lrs, train_loss_list, valid_loss_list

    def one_fold_training(self, X_train, y_train, X_val, y_val, country='DE'):
        # create model
        assert self.modelType in ['VariationalLSTM']
        if self.modelType == 'VariationalLSTM':
            model = VariationalLSTM(num_neurons = self.num_neurons, input_window_size = self.input_window_size, predicted_step = self.predicted_step, device = self.device)
        # create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # employ learning rate scheduler, the default scheduler wonnt change the learning rate, feel free to modify it
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
        # create loss function
        criterion = None
        assert self.lossType in ['qd','winkler']
        if self.lossType == 'qd':
            criterion = qd_objective(lambda_=self.lambda_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        elif self.lossType == 'winkler':
            criterion = winkler_objective(lambda_=self.lambda_, alpha_=self.alpha_, soften_=self.soften_, device=self.device, batch_size=self.batch_size)
        # to device
        model = model.to(self.device)
        criterion = criterion.to(self.device) 
        
        # begin training
        lrs, train_loss_list, valid_loss_list= self.training_loop(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, model = model, optimizer = optimizer, scheduler = scheduler, criterion = criterion)

        if self.draw:
            prefix = country + '/' + self.modelType + '_fold' + str(self.fold_size)+'_'

            # Convergence grapha
            plt.plot(train_loss_list[:], color='r', linewidth=3, label='Training loss')
            plt.plot(valid_loss_list[:], color='b', linewidth=3, label='Validation loss')
            plt.xlabel("Training epoches",fontsize=64)
            plt.ylabel("Training loss",fontsize=64)
            plt.xticks(fontsize = 42)
            plt.yticks(fontsize = 42)
            plt.legend(loc="upper left",fontsize=32)
            # plt.savefig('./fig/'+prefix+'loss'+'.png')
            plt.show()

            # plot and view some predictions
            predictions = []
            for t in range(self.num_forward_passes):
                predictions.append(model(Variable(torch.tensor(X_val[300:300+self.display_size],dtype=torch.float)).to(self.device)).cpu().detach().numpy())
            pred_array = np.array(predictions)
            y_u_pred = pred_array[:,:,0]
            y_l_pred = pred_array[:,:,1]
            y_u_mean = y_u_pred.mean(axis=0)
            y_u_std = y_u_pred.std(axis=0)
            y_l_mean = y_l_pred.mean(axis=0)
            y_l_std = y_l_pred.std(axis=0)

            n_std_devs = 1.96
            if self.alpha_ == 0.05:
                n_std_devs = 1.96
            elif self.alpha_ == 0.10:
                n_std_devs = 1.645
            elif self.alpha_ == 0.01:
                n_std_devs = 2.575
            
            # from top to bottom, the upper bound of the upper bound, the lower bound of the upper bound, the upper bound of the lower bound, the lower bound of the lower bound, of the constructed PI 
            y_u_u = y_u_mean+n_std_devs*y_u_std
            y_u_l = y_u_mean-n_std_devs*y_u_std
            y_l_u = y_l_mean+n_std_devs*y_l_std
            y_l_l = y_l_mean-n_std_devs*y_l_std

            plt.plot(np.arange(self.display_size),y_val[300:300+self.display_size,0],linewidth=3, color='black',label='Observations')

            plt.plot(np.arange(self.display_size), y_u_mean, color='g') # upper boundary prediction
            plt.plot(np.arange(self.display_size), y_l_mean, color='g') # lower boundary prediction
            plt.fill_between(np.arange(self.display_size), y_u_mean, y_l_mean, color='g', alpha=0.5,label='Aleatoric Uncertainty')

            plt.plot(np.arange(self.display_size), y_u_u, color='b') 
            plt.plot(np.arange(self.display_size), y_u_l, color='b')
            plt.fill_between(np.arange(self.display_size), y_u_u, y_u_l, color='b', alpha=0.3,label='Epistemic Uncertainty of Upper Bounds')

            plt.plot(np.arange(self.display_size), y_l_u, color='r') 
            plt.plot(np.arange(self.display_size), y_l_l, color='r')
            plt.fill_between(np.arange(self.display_size), y_l_u, y_l_l, color='r', alpha=0.3,label='Epistemic Uncertainty of Lower Bounds')
            plt.xlabel("Time(hour)",fontsize=64)
            plt.ylabel("Normalized wind power",fontsize=64)
            plt.xticks(fontsize = 42)
            plt.yticks(fontsize = 42)
            plt.legend(loc="upper left",fontsize=32)

            # plt.savefig('./fig/'+prefix+'PIs.png')
            plt.show()
        # print some stats
        predictions = []
        for t in range(self.num_forward_passes):
            prediction = None
            for batch in range(math.ceil(X_val.shape[0]/self.batch_size)):
                start = batch * self.batch_size
                end = start + self.batch_size
                if prediction is None: 
                    prediction = model(Variable(torch.tensor(X_val[start:end],dtype=torch.float)).to(self.device)).cpu().detach().numpy()
                else:
                    prediction = np.concatenate((prediction, model(Variable(torch.tensor(X_val[start:end],dtype=torch.float)).to(self.device)).cpu().detach().numpy()), axis=0)
            predictions.append(prediction)
        pred_array = np.array(predictions)
        y_u_pred = pred_array[:,:,0]
        y_l_pred = pred_array[:,:,1]
        y_u_mean = y_u_pred.mean(axis=0)
        y_u_std = y_u_pred.std(axis=0)
        y_l_mean = y_l_pred.mean(axis=0)
        y_l_std = y_l_pred.std(axis=0)

        K_u = np.maximum(0.0, np.sign(y_u_mean - y_val[:,0]))
        K_l = np.maximum(0.0, np.sign(y_val[:,0] - y_l_mean))
        picp = np.mean(K_u * K_l)
        mpiw = np.mean(np.absolute(y_u_mean - y_l_mean))
        S_t = np.absolute(y_u_mean-y_l_mean) + (2/self.alpha_)*(np.multiply((y_l_mean-y_val[:,0]), np.maximum(0.0, np.sign(y_l_mean - y_val[:,0])))) + (2/self.alpha_)*(np.multiply(y_val[:,0]-y_u_mean, np.maximum(0.0,np.sign(y_val[:,0] - y_u_mean))))
        S_overline = np.mean(S_t)
        print('PICP:', picp)
        print('MPIW:', mpiw)
        print('Winkler Score:',S_overline)
        print('Upper Bound Mean:', np.mean(y_u_mean))
        print('Upper Bound Std:', np.mean(y_u_std))
        print('Lower Bound Mean:', np.mean(y_l_mean))
        print('Lower Bound Std:', np.mean(y_l_std))
        return picp, mpiw, S_overline, np.mean(y_u_mean), np.mean(y_u_std), np.mean(y_l_mean), np.mean(y_l_std)
    
    def run(self, country='DE'):
        # create train, test dataset
        X_train_list, y_train_list, X_val_list, y_val_list = self.train_test_split(country=country)
        picp_list = []
        mpiw_list = []
        ace_list = []
        score_list = []
        bayesian_dict = {}
        for key in ['ubm', 'ubs', 'lbm', 'lbs']:
            bayesian_dict[key] = []
        for i in range(self.fold_size):
            print('--------------------------------------------------fold'+str(i)+'---------------------------------------------------------')
            picp, mpiw, score, ubm, ubs, lbm, lbs = self.one_fold_training(X_train_list[i], y_train_list[i], X_val_list[i], y_val_list[i], country = country)
            picp_list.append(picp)
            mpiw_list.append(mpiw)  
            ace_list.append(picp-(1-self.alpha_))
            score_list.append(score) 
            bayesian_dict['ubm'].append(ubm)
            bayesian_dict['ubs'].append(ubs)
            bayesian_dict['lbm'].append(lbm)
            bayesian_dict['lbs'].append(lbs)
        print("picp mean: "+str(np.mean(picp_list)))
        print("picp std: "+str(np.std(picp_list)))
        print("mpiw mean: "+str(np.mean(mpiw_list)))
        print("mpiw std: "+str(np.std(mpiw_list)))
        print("ace mean: "+str(np.mean(ace_list)))
        print("ace std: "+str(np.std(ace_list)))
        print("winkler score mean: "+str(np.mean(score_list)))
        print("winkler score std: "+str(np.std(score_list)))
        print("mean of upper bound mean: "+str(np.mean(bayesian_dict['ubm'])))
        print("std of upper bound mean: "+str(np.std(bayesian_dict['ubm'])))
        print("mean of upper bound std : "+str(np.mean(bayesian_dict['ubs'])))
        print("std of upper bound std : "+str(np.std(bayesian_dict['ubs'])))
        print("mean of lower bound mean: "+str(np.mean(bayesian_dict['lbm'])))
        print("std of lower bound mean: "+str(np.std(bayesian_dict['lbm'])))
        print("mean of lower bound std: "+str(np.mean(bayesian_dict['lbs'])))
        print("std of lower bound std: "+str(np.std(bayesian_dict['lbs'])))
