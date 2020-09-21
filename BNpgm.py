import pandas as pd
import bnlearn
from sklearn.model_selection import train_test_split

print("uri yallow is in this bitch !")

df = pd.read_csv(r"movies_pgm.csv")
df = df[df['budget'] > 10000]
df = df[df['top actor'].notnull() & df['top director'].notnull()]
df_final = df[['budget scale', 'top actor', 'top director', 'label']]
df_train, df_test = train_test_split(df_final, test_size=0.2, random_state=0)


edges = [('top actor', 'budget scale'),
         ('top director', 'budget scale'),
         ('budget scale', 'label'),
]

# Make the actual Bayesian DAG
DAG = bnlearn.make_DAG(edges)

# DAG is stored in adjacency matrix
print(DAG['adjmat'])

# # No CPDs are in the DAG. Lets see what happens if we print it.
# bnlearn.print_CPD(DAG)

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
    prob = bnlearn.inference.fit(DAG, variables=['label'], evidence={'top actor':r['top actor'],
                                                                     'top director':r['top director'],
                                                                     'budget scale':r['budget scale']})
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

