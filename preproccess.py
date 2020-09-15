

import pandas as pd
import numpy as np
from time import time
from os import listdir
from datetime import datetime

ORIG_FILE = 'data_labeled.csv'
VEC_PATH = 'data/'
VEC_FILTERED_PATH = 'data_filtered/'
BUDGET_THRESHOLD = 10000

COLUMNS_TO_DROP = ['belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'popularity',
                     'poster_path', 'status']

COLS_FOR_MODEL_2 = ['budget', 'genres', 'original_language', 'production_companies', 'success', 'release_date',
                    'runtime']

COLS_TO_VEC = {'genres': '\'name\':',
               'production_counries': '\'iso_3166_1\':'}


class Dataset:
    def __init__(self,
                 orig_file=True,
                 vectorized='all',
                 vectorized_filtered=0):
        self.orig_df, self.vec_df, self.vec_df_filtered = self.load_df(orig_file, vectorized, vectorized_filtered)
        df = self.vec_df_filtered if self.vec_df_filtered is not None else self.vec_df
        self.vocab = {}
        # data_X, data_y = df.drop('success', axis=1).to_numpy(), df['success'].values


    def load_df(self, orig_file, vectorized, vectorized_filtered):
        if orig_file:
            orig_df = pd.read_csv(ORIG_FILE, index_col='id')
            orig_df = orig_df.drop(COLUMNS_TO_DROP, axis=1, errors='ignore')
            print(f"original file loaded")
        else:
            orig_df = None
        vec_df = self.load_vec(vectorized, 'vectorized', VEC_PATH)
        vec_filtered_df = self.load_vec(vectorized_filtered, 'vectorized_filtered', VEC_FILTERED_PATH)
        return orig_df, vec_df, vec_filtered_df

    @ staticmethod
    def load_vec(file_num, name, path):
        if file_num == 'all':
            file_num = len(listdir(path))
        if file_num == 0:
            return None
        print(f"Loading {file_num} {name} files")
        file_list = sorted(listdir(path))[:file_num]
        vec_df_list = []
        for i, f in enumerate(file_list):
            t = time()
            print(f"loading file {name}_{i}... start at {datetime.now()}")
            vec_df_list.append(pd.read_csv(path + f, index_col='id'))
            print(f"file {name}_{i}.csv has loaded in {round(time()-t, 3)} sec")
        t = time()
        print(f"concatinadting... start at {datetime.now()}")
        vec_df = pd.concat(vec_df_list)
        print(f"finished concatinating {file_num} {name} files in {round(time() - t, 3)} sec")
        return vec_df

    @staticmethod
    def save_to_subfiles(df: pd.DataFrame, output_folder, name, num=10):
        part = int(len(df.index) / num)
        idx_list = [list(range(i * part, (i+1) * part)) for i in range(num)]
        for i, idx in enumerate(idx_list):
            print(f"saving file {i}")
            t = time()
            tmp = df.iloc[idx]
            tmp.to_csv(output_folder + name + '_' + str(i) + '.csv')
            print(f"file {i} saved in {round(time() - t, 3)} sec")

    def normalize(self):
        df = self.vec_df_filtered.copy()
        m = max(self.vocab.values()) + 1
        for col in df:
            if col == 'success':
                continue
            df[col] = [(x / m) ** 0.5 for x in df[col].values]
        return df


    def orig_to_numbered(self, filter=True):
        df = self.orig_df.copy()
        if filter:
            df = df[df['budget'] > BUDGET_THRESHOLD]
        rev, bud = df['revenue'].values, df['budget'].values
        df['success'] = [1 if rev[i] > bud[i] else 0 for i in range(len(rev))]
        df = df.drop('revenue', axis=1, errors='ignore')
        df = df[COLS_FOR_MODEL_2]
        df['day_of_year'] = [datetime.strptime(x, '%m/%d/%y').timetuple().tm_yday for x in df['release_date'].values]
        df['year'] = [datetime.strptime(x, '%m/%d/%y').timetuple().tm_year for x in df['release_date'].values]
        df = df.drop('release_date', axis=1)
        print(df['day_of_year'])
        print(df['year'])
        vocab = {}
        i = 2
        for col in df.columns:
            if col == 'success':
                continue
            print(col)
            new_col = []
            for val in df[col].values:
                if col in COLS_TO_VEC:
                    try:
                        val = val.replace(' ', '').split(COLS_TO_VEC[col])[1:]
                        val = tuple([x.split('\'')[1] for x in val])
                        # print(val)
                    except:
                        val = -1
                    # print(val)
                if val not in vocab:
                    vocab[val] = i
                    i += 1
                new_col.append(vocab[val])
            df[col] = new_col

        self.vocab = vocab
        return df[[x for x in list(df.columns.values) if x != 'success'] + ['success']]




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
# d = Dataset(False, 'all')
