import pandas as pd
import numpy as np

from tqdm import tqdm
from exceptions import FileNotFound
from os import path
from tensorflow.keras.utils import to_categorical
from queue import Queue

REDUCTION_FACTOR = 0.7

####################
# Static functions #
####################


def data_classification(X, n_input):
    [N, D] = X.shape # returns the dimension of the set. N_Observations x 40
    dataX = np.zeros((N - n_input + 1, n_input, D)) # Create N-H + 1 matrixes of n_input x 40 dimension
    print("......Preparing input data for the network......")
    for i in tqdm(range(n_input, N + 1)):
        dataX[i - n_input] = X[i - n_input:i, :]
    print("......Data loaded......")
    return dataX.reshape(dataX.shape + (1,))


def prepare_x_y(data, n_input):
    x = np.array(data.iloc[:, 2:-2])
    y = data.iloc[:, -1:].values
    y = y[n_input -1:]
    x = data_classification(x, n_input)
    y = to_categorical(y, 3)
    return x, y


def load_df(rel_path, reduce=True):
    try:
        if not path.exists(rel_path):
            raise FileNotFound(f"No file for : {rel_path}")
        print('.....Loading df......')
        df = pd.read_csv(rel_path).set_index('Matching Time')
        print(f"Df loaded from source: {rel_path.split('/')[-1]}")
        if reduce:
            df = df[:int(len(df) * REDUCTION_FACTOR)]
        return df
    except FileNotFound as err:
        print(err)
        return None


def create_queue_from_delta_series(delta_series):
    queue = Queue(maxsize=len(delta_series))
    for index, element in enumerate(delta_series):
        queue.put((index, element))
    return queue
