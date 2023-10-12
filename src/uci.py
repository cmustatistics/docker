# Read and prepare dataset for UCI experiments
import pandas as pd
import numpy as np
from dotdic import DotDic
from loader import save_to

def normalize(X):

    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]

    return (X - X_mean) / X_std, X_mean, X_std

def preprocess(X, Y):

        X, X_mean, X_std = normalize(X)
        Y, Y_mean, Y_std = normalize(Y)

        return X, Y, X_mean, X_std, Y_mean, Y_std

def prepare(str, seed=123, prop=0.8):

    np.random.seed(seed)

    local_base_folder = './data_source/'

    if str=="protein":
        # Protein data https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv
        data_url = local_base_folder + 'protein/CASP.csv'
        raw_data = pd.read_csv(data_url)

        X_raw = np.array(raw_data.iloc[ : , 1:])
        Y_raw = np.array(raw_data.iloc[:, 0])[:,None]

    elif str=="bike":
        # Bike data https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
        data_dir = local_base_folder+'bike_sharing/hour.csv'
        raw_data = pd.read_csv(data_dir)

        X_raw = np.array(raw_data.iloc[:,2:-3])
        Y_raw = np.array(raw_data.iloc[:,-1])[:,None]
        Y_raw = np.log(Y_raw)

    elif str=="airlines":
        data_dir = local_base_folder+'airlines/airlines.csv'
        raw_data = pd.read_csv(data_dir)

        X_raw = np.array(raw_data.iloc[:,0:8])
        Y_raw = np.array(raw_data.iloc[:,8])[:,None]

    elif str=="song":
        # Song data (to unzip) https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
        data_dir = local_base_folder+'song/YearPredictionMSD.txt'
        raw_data = pd.read_csv(data_dir)


        X_raw = np.array(raw_data.iloc[:,1:90])
        Y_raw = np.array(raw_data.iloc[:,0])[:,None]
        Y_raw=np.log(2015-Y_raw)

    elif str=="sgemm":
        # sgemm (to unzip) https://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip

        data_dir = local_base_folder+'sgemm/sgemm_product.csv'
        raw_data = pd.read_csv(data_dir)
        raw_data = pd.DataFrame(raw_data)

        raw_data['average'] = raw_data.iloc[:, 14:].astype(float).mean(axis=1)

        X_raw = np.array(raw_data.drop(raw_data.columns[14:],axis=1))
        Y_raw = np.array(raw_data.iloc[:,18])[:,None]
        Y_raw=np.log(Y_raw)

    elif str=="gas":
        # Gas sensor (to unzip) https://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip

        data_dir = local_base_folder+'gas/ethylene_CO.txt'
        raw_data = pd.read_csv(data_dir,skiprows=1,delim_whitespace=True,header=None)
        raw_data = pd.DataFrame(raw_data)

        raw_data=raw_data[raw_data[1]>0]

        X_raw = np.array(raw_data.drop([1,2],axis=1))
        Y_raw = np.array(raw_data.iloc[:,1])[:,None]

    elif str=="buzz":
        # Buzz (to unzip) https://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz

        data_dir = local_base_folder+'buzz/regression/Twitter/Twitter.data'
        raw_data = pd.read_csv(data_dir,header=None)
        raw_data = pd.DataFrame(raw_data)

        X_raw = np.array(raw_data.drop([77],axis=1))
        Y_raw = np.array(raw_data.iloc[:,77])[:,None]
        Y_raw = np.log(1+Y_raw)

    else:
        raise ValueError('Dataset not available')

    X, Y, X_mean, X_std, Y_mean, Y_std = preprocess(X_raw, Y_raw)
    N = X.shape[0]

    ind = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(ind)

    n = int(N * prop)
    inTestMax = int(np.min([N, n + 100000]))

    data = DotDic({
        'Xtrain': X[ind[:n]],
        'Xtest': X[ind[n:inTestMax]],
        'Ytrain': Y[ind[:n]],
        'Ytest': Y[ind[n:inTestMax]],
        'D': X.shape[1],
        'Ystd': Y_std
    })

    print("Saving {0} dataset Ntrain = {1} Ntest = {2} D = {3}"
          .format(str, data.Xtrain.shape[0], data.Xtest.shape[0], data.D))

    return data

def save_data(str, seed=123, prop=0.8):
    save_to(obj=prepare(str=str, seed=seed, prop=prop),
            object_str='data',
            path='./data/{0}'.format(str))
