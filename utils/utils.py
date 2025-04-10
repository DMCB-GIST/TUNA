import numpy as np
import torch
import pickle
import sklearn.metrics as m
from scipy.stats import pearsonr

def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_sparse(file_name, dtype):
    contactmap = np.load(file_name + '.npy', allow_pickle=True)
    contactmap = [d.tocoo() for d in contactmap]
    return [dtype(torch.LongTensor(np.vstack((d.row.tolist(),d.col.tolist()))), torch.FloatTensor(d.data), torch.Size(d.shape)) for d in contactmap]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def load_pickle_multi(file_name):
    with open(file_name, 'rb') as f:
        pro_id = pickle.load(f)
        com_id = pickle.load(f)
        return (pro_id, com_id)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def c_index(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair is not 0:
        return summ / pair
    else:
        return 0


def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred,y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))

def multi_slope_cal(input):
    order = [1,2,3,4,5,6,7,8,9,10]
    length = len(order)
    sum_x = sum(order)
    sum_y = sum(input)
    sum_xy = np.sum(np.array(order)*np.array(input))
    sum_xx = np.sum(np.array(order)**2)
    
    slope = (length * sum_xy - (sum_x * sum_y)) / (length * sum_xx - (sum_x)**2)
    
    return slope
