import pandas as pd
import bnlearn
from ast import literal_eval
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"movies_pgm.csv")
df = df[df['budget'] > 10000]
# df = df[df['top actor'].notnull() & df['top director'].notnull()]

df.loc[df['popularity'] <= 5, 'pop scale'] = 1
df.loc[(df['popularity'] > 5)&(df['popularity'] <= 10), 'pop scale'] = 2
df.loc[(df['popularity'] > 10)&(df['popularity'] <= 50), 'pop scale'] = 3
df.loc[(df['popularity'] > 50)&(df['popularity'] <= 100), 'pop scale'] = 4
df.loc[df['popularity'] > 100, 'pop scale'] = 5

df['main genre'] = None
for index, r in df.iterrows():
    if not r['genres']:
        continue
    try:
        row = literal_eval(r['genres'])
    except ValueError:
        continue
    g = row[0]['name']
    df.loc[index, 'main genre'] = g

df = df[df['top actor'].notnull() & df['top director'].notnull() & df['budget scale'].notnull()
        & df['main genre'].notnull() & df['pop scale'].notnull()]

df_final = df[['top director', 'budget scale', 'week_num', 'main genre', 'top actor', 'pop scale', 'label']]

df_train, df_test = train_test_split(df_final, test_size=0.2, random_state=0)


edges = [('top director', 'budget scale'),
         ('top director', 'week_num'),
         ('top director', 'main genre'),
         ('budget scale', 'label'),
         ('week_num', 'pop scale'),
         ('main genre', 'top actor'),
         ('top actor', 'pop scale'),
         ('label', 'top actor'),
]

# Make the actual Bayesian DAG
DAG = bnlearn.make_DAG(edges)

# DAG is stored in adjacency matrix
print(DAG['adjmat'])

# No CPDs are in the DAG. Lets see what happens if we print it.
bnlearn.print_CPD(DAG)

# # Plot DAG. Note that it can be differently orientated if you re-make the plot.
# print("plotting")
# bnlearn.plot(DAG)
# print("stop plot")

# Parameter learning on the user-defined DAG and input data
DAG = bnlearn.parameter_learning.fit(DAG, df_train)

# Print the learned CPDs
bnlearn.print_CPD(DAG)


# test set
c = 0
size = 0
for index, r in df_test.iterrows():
    prob = bnlearn.inference.fit(DAG, variables=['label'], evidence={'top director': r['top director'],
                                                                     'week_num': r['week_num'],
                                                                     'main genre': r['main genre'],
                                                                     'top actor': r['top actor'],})
    score = prob.values[1]
    real_label = r['label']
    if score > 0.6:
        predict = 1
    else:
        predict = 0
    if real_label == predict:
        c += 1
    size += 1
    # print(r)
    # print(f"index: {index}\t probability to success: {prob.values[1]}\t true label: {r['label']}")

print(f"\n\n\naccuracy is:{c/size}")
