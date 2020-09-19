import pandas as pd
from ast import literal_eval
#import bnlearn
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"movies_pgm.csv")

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

# No CPDs are in the DAG. Lets see what happens if we print it.
bnlearn.print_CPD(DAG)

# Plot DAG. Note that it can be differently orientated if you re-make the plot.
bnlearn.plot(DAG)

# Parameter learning on the user-defined DAG and input data
DAG = bnlearn.parameter_learning.fit(DAG, df_train, methodtype='maximumlikelihood')

# Print the learned CPDs
bnlearn.print_CPD(DAG)

# test set
for index, r in df_train.iterrows():
    prob = bnlearn.inference.fit(DAG, variables=['label'], evidence={'top actor':r['top actor'],
                                                                     'top director':r['top director'],
                                                                     'budget scale':r['budget scale']})
    print(f"index: {index}\tprobability to 1: {prob[1]}\ttrue label: {r['label']}")


