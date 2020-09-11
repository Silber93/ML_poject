

import pandas as pd
import numpy as np
from time import time
from os import listdir


ORIG_FILE = 'train.csv'
VEC_PATH = 'data/'
VEC_FILTERED_PATH = 'data_filtered'



class Dataset:
    def __init__(self,
                 orig_file=True,
                 vectorized=None,
                 vectorized_filtered=0):
        print(vectorized)
        self.orig_df, self.vec_df, self.vec_df_filtered = self.load_df(orig_file, vectorized, vectorized_filtered)

    def load_df(self, orig_file, vectorized, vectorized_filtered):
        if orig_file:
            orig_df = pd.read_csv(ORIG_FILE)
            print(f"original file loaded")
        else:
            orig_df = None
        vec_df = self.load_vec(vectorized, 'vectorized')
        vec_filtered_df = self.load_vec(vectorized_filtered, 'vectorized_filtered')
        return orig_df, vec_df, vec_filtered_df

    @ staticmethod
    def load_vec(file_num, name):
        print(file_num)
        if file_num == 'all':
            file_num = len(listdir(VEC_PATH))
        if file_num == 0:
            return None
        print(f"Loading {file_num} {name} files")
        file_list = sorted(listdir(VEC_PATH))
        t = time()
        vec_df_list = [pd.read_csv(VEC_PATH + file_list[i]) for i in range(file_num)]
        vec_df = pd.concat(vec_df_list)
        print(f"{file_num} {name} files have been loaded in {round(time() - t), 3} sec")
        return vec_df






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


# cols_to_drop = ['id', 'overview', 'original_language', 'tagline', 'title', 'Unnamed: 0']
# filepath = 'dummies.csv'
# print("loading data")
# t = time()
# data = pd.read_csv(filepath)
# print(f"loaded in {round(time() - t, 3)} sec")
#
# divide_to_subfiles(data)
d = Dataset(False, 'all')
