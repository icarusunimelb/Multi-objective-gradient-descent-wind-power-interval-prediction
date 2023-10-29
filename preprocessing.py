import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

DATASET_PATH = './dataset/ninja_wind_europe_v1.1_current_national.csv'
INPUT_WINDOW_SIZE = 24
PREDICTED_STEP = 6

def data_reader():
    df = pd.read_csv(DATASET_PATH)
    df = df.set_index('time')
    # Select four countries as samples
    # DE - Germany, ES - Spain, FR - France, GB - Great Britain, IT - Italy, CH - Switzerland, SE - Sweden, DK - Denmark, NO - Norway, LU - Luxembourg
    # selected_countries = ['DE', 'ES', 'FR', 'GB', 'IT', 'CH', 'SE', 'DK', 'NO', 'LU']
    selected_countries = ['DE', 'ES', 'FR', 'GB']
    selected_df = df[selected_countries]
    recent_selected_df = selected_df['2014-01-01 00:00:00': '2016-12-31 23:00:00']
    return recent_selected_df

# transfer time series data to supervised data
# Shift the real value to the next time stamp to make the prediction work
# Formulate the sequence of input and output
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # output(t, t+1, ... t+n_out)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # concat the columns 
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop the rows without t-i data
    if dropnan:
       agg.dropna(inplace=True) 
    return agg   

def pre_process(data, country='DE', input_window_size=48, predicted_step=1):
    # normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_np = scaler.fit_transform(data[country].values.reshape(-1, 1))
    
    supervised_df = series_to_supervised(scaled_np, input_window_size, predicted_step)
    
    supervised_df_path = './dataset/'+str(predicted_step)+'_'+country+'_supervised_wind_power.csv'
    supervised_df.to_csv(supervised_df_path, index_label='index')

df = data_reader()
for country in df.columns:
    pre_process(df, country=country, input_window_size=INPUT_WINDOW_SIZE, predicted_step=PREDICTED_STEP) 