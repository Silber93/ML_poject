from preproccess import Dataset, VEC_PATH, VEC_FILTERED_PATH
from models import *
from random import sample
feature_sets = \
    [['cast', 'crew'],
     ['genres', 'production_countries'],
     ['production_companies', 'budget']]

feature_sets_2 = \
    [['budget'], ['day_of_year'], ['year']]


# d = Dataset(orig_file=False, vectorized=0, vectorized_filtered=1)
d = Dataset(orig_file=True, vectorized=0, vectorized_filtered=0)
d.vec_df_filtered = d.orig_to_numbered()
df = d.normalize()
d.vec_df_filtered = df
r = model_2(d, feature_sets=[[x] for x in d.vec_df_filtered.columns if x != 'success'])
r.run()
# X = d.vec_df_filtered[[x[0] for x in feature_sets_2]]
# y = d.vec_df_filtered['success']
# print(X)
# print(y)
# print(d.vec_df_filtered)
# r = model_2(d, feature_sets_2)
# r.run()
# print("hi")
# r = RNN_model_test_2()

