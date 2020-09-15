import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from preproccess import Dataset
from random import sample, randint
from matplotlib import pyplot as plt
class RNN_cell:
    def __init__(self):
        print("hello")


class model_2:
    def __init__(self, dataset: Dataset, feature_sets, p=True):
        self.p = p
        self.dataset = dataset
        self.feature_sets = feature_sets
        self.raw_data = dataset.vec_df_filtered if dataset.vec_df_filtered is not None else dataset.vec_df


    def organize_input(self):
        inputs = []
        for set in self.feature_sets:
            cols = []
            for col in set:
                cols += [x for x in list(self.raw_data.columns.values) if col in x]
            inputs.append(self.raw_data[cols])
        return inputs

    def train_test_split(self, inputs, train_size=0.8):
        train_size = int(self.raw_data.shape[0] * train_size)
        all_idx = list(range(self.raw_data.shape[0]))
        train_idx = sample(all_idx, train_size)
        test_idx = [x for x in all_idx if x not in train_idx]
        X_train = [x.iloc[train_idx].to_numpy() for x in inputs]
        X_test = [x.iloc[test_idx].to_numpy() for x in inputs]
        y_train = self.raw_data.iloc[train_idx]['success'].values
        y_test = self.raw_data.iloc[test_idx]['success'].values
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def run(self):
        print(max(self.dataset.vocab.values()))
        if self.p: print("model 2 running")
        inputs = self.organize_input()
        X = self.raw_data.drop('success', axis=1).to_numpy()
        X = np.array([[[x] for x in row] for row in X], dtype=float)
        y = self.raw_data['success'].to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # X_train, X_test, y_train, y_test = self.train_test_split(inputs)
        print(X_train.shape)
        rnn_model = tf.keras.models.Sequential()
        # rnn_model.add(tf.keras.layers.Embedding(1000, 10))
        rnn_model.add(tf.keras.layers.SimpleRNN(7))
        rnn_model.add(tf.keras.layers.Dense(1))
        rnn_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        # rnn_model.summary()
        history = rnn_model.fit(X_train, y_train, epochs=100)

        res = rnn_model.predict(X_test)
        print(res)
        self.plot_results(history, res, y_test)

    def plot_results(self, history, res, y_test):
        print(len(res), len(y_test))
        plt.scatter(range(len(res)), res, c='r')
        plt.scatter(range(len(y_test)), y_test, c='g')
        plt.show()
        plt.plot(history.history['loss'])
        plt.show()


class RNN_model_test:
    def __init__(self):
        path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        print(text[:13])
        vocab = list(dict.fromkeys(sorted(text)))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        text_as_int = np.array([char2idx[c] for c in text])
        print('{')
        for char, _ in zip(char2idx, range(20)):
            print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('  ...\n}')

        print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
        seq_length = 100
        examples_per_epoch = len(text) // (seq_length + 1)

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        for item in sequences.take(5):
            s = repr(''.join(idx2char[item.numpy()]))
            print(s, len(s))
        print()

        dataset = sequences.map(self.split_input_target)

        for input_example, target_example in dataset.take(1):
            print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
            print('Target data:', repr(''.join(idx2char[target_example.numpy()])))
        print()

        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
        print()
        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        print(dataset, '\n')

        # Length of the vocabulary in chars
        vocab_size = len(vocab)

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        model = self.build_model(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            batch_size=BATCH_SIZE)

        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)
            print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

        model.summary()

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

        print(sampled_indices)

        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    @ staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    @ staticmethod
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text


class RNN_model_test_2:
    def __init__(self):
        data = [[[(i + j)] for i in range(2, 7)] for j in range(100)]
        target = [1 if any([sum([y[0] for y in x]) % j == 0 for j in [3, 4]]) else 0 for x in data]
        data = np.array(data, dtype=float)
        target = np.array(target, dtype=float)
        print(data)
        print(target)
        print(data.shape)
        print(target.shape)
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
        m = tf.keras.models.Sequential()
        m.add(tf.keras.layers.Embedding(1000, 5))
        m.add(tf.keras.layers.SimpleRNN(1, batch_input_shape=(None, 5, 1), return_sequences=True))
        m.add(tf.keras.layers.SimpleRNN(1, return_sequences=False))
        m.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        m.summary()
        history = m.fit(x_train, y_train, epochs=500)
        res = m.predict(x_test)
        res = [1 if x >= 0.5 else 0 for x in res]
        print(res)
        plt.scatter(range(20), res, c='r')
        plt.scatter(range(20), y_test, c='g')
        plt.show()
        plt.plot(history.history['loss'])
        plt.show()
