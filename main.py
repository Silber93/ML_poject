from preproccess import Dataset, VEC_PATH, VEC_FILTERED_PATH
from models import *

d = Dataset(orig_file=False, vectorized='all')
budget = d.vec_df['budget'].values
revenue = d.vec_df['revenue'].values
d.vec_df['success'] = [1 if revenue[i] > budget[i] else 0 for i in range(len(budget))]
tmp = d.vec_df.drop('revenue', axis=1)
ll = list(tmp.columns.values)
print(len(ll))
print(ll[-1])
print([i for i in range(len(ll)) if ll[i] == 'success'])
d.divide_to_subfiles(tmp, VEC_PATH, 'vectorized')
d.vec_df = tmp
d.filter_budget(10000)

