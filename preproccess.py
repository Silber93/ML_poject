

import pandas as pd
import numpy as np
from time import time


def divide_to_subfiles(df: pd.DataFrame, num=10):
    part = int(len(df.index) / num)
    idx_list = [list(range(i * part, (i+1) * part)) for i in range(num)]
    print(idx_list)
    for i, idx in enumerate(idx_list):
        print(f"saving file {i}")
        t = time()
        tmp = df.iloc[idx]
        tmp.to_csv('data/dummies_' + str(i) + '.csv')
        print(f"file {i} saved in {round(time() - t, 3)} sec")

def calc_b(X, Y, b_only=False):
    print("transopising")
    t = time()
    Xt = X.transpose()
    print(f"transpose finished in {time() - t}")
    XtY = Xt.dot(Y)
    XtX = Xt.dot(X)
    inv_matrix = np.linalg.pinv(XtX)
    b = inv_matrix.dot(XtY)
    if b_only:
        return b
    return b, inv_matrix

def get_b_inv(df, b_only=True):
    X = df.drop(['budget'], axis=1)
    X.insert(0, 'ones', 1)
    X = X.to_numpy(dtype='float')
    Y = df['budget'].to_numpy(dtype='float')
    return calc_b(X, Y, b_only)


def rearrange(data):
    cast_cols = [x for x in data.columns if "cast" in x]
    crew_cols = [x for x in data.columns if "crew" in x]
    cast = data[cast_cols]
    crew = data[crew_cols]

    n0 = len(data.index)
    data = data.set_index('id')
    data = data.drop(cols_to_drop, axis=1, errors='ignore')
    data = data.drop(cast_cols + crew_cols, axis=1, errors='ignore')
    seen_data = data[data['budget'] > 10000]

    print(f"data size: {seen_data.shape}")
    print(f"getting b...")
    t = time()
    b = get_b_inv(seen_data)
    print(f"completed in {round(time() - t, 3)} sec")


cols_to_drop = ['id', 'overview', 'original_language', 'tagline', 'title', 'Unnamed: 0']
filepath = 'dummies.csv'
print("loading data")
t = time()
data = pd.read_csv(filepath)
print(f"loaded in {round(time() - t, 3)} sec")

divide_to_subfiles(data)

