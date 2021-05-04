import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import entropy


data_root = os.path.join(os.path.dirname(__file__), 'data')
RANDOM_STATE = 7


# Source: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def main():
    data = pd.read_csv(os.path.join(data_root, 'cardio', 'cardio_train.csv'), sep=';')
    x, y = data.drop(columns=['cardio', 'id']), data['cardio']
    # Split the data into train, validation and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Shuffle the train set (probably unnecessary, but good to do since we will later split the samples by index)
    x_train, y_train = shuffle_in_unison(x_train, y_train)

    # 5% of the samples in the train set are actually labeled to start with
    known_samples = int(0.05 * len(x_train))

    model = LogisticRegression()

    accuracies = np.zeros(len(y_train) - known_samples + 1)
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    for i in range(len(accuracies)):
        print(i)
        # model.fit(x_train_l, y_train_l)
        model.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = model.predict(x_val)
        accuracies[i] = accuracy_score(y_val, y_pred)
        precisions[i] = precision_score(y_val, y_pred)
        recalls[i] = recall_score(y_val, y_pred)
        f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):

            probabilities = model.predict_proba(x_train[known_samples:])

            entropies = [entropy(probabilities[j], base=2) for j in range(len(probabilities))]

            most_unsure = np.argmax(entropies)

            x_train[[most_unsure, known_samples]] = x_train[[known_samples, most_unsure]]
            y_train[[most_unsure, known_samples]] = y_train[[known_samples, most_unsure]]
            # x_train[most_unsure], x_train[known_samples] = x_train[known_samples], x_train[most_unsure]
            # y_train[most_unsure], y_train[known_samples] = y_train[known_samples], y_train[most_unsure]
            known_samples += 1

            # x_most_unsure = x_train_u[most_unsure]
            # y_most_unsure = y_train_u[most_unsure]
            #
            # x_train_l = np.append(x_train_l, x_most_unsure)
            # y_train_l = np.append(y_train_l, y_most_unsure)
            #
            # x_train_u = np.delete(x_train_u, most_unsure)
            # y_train_u = np.delete(y_train_u, most_unsure)

    random_accuracies = np.zeros_like(accuracies)
    random_precisions = np.zeros_like(accuracies)
    random_recalls = np.zeros_like(accuracies)
    random_f1s = np.zeros_like(accuracies)

    known_samples = int(0.05 * len(x_train))

    model = LogisticRegression()

    for i in range(len(random_accuracies)):
        print(i)
        model.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = model.predict(x_val)
        random_accuracies[i] = accuracy_score(y_val, y_pred)
        random_precisions[i] = precision_score(y_val, y_pred)
        random_recalls[i] = recall_score(y_val, y_pred)
        random_f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):
            new_known = np.random.randint(known_samples, len(x_train))  # TODO: Check this again (random number from the unknown indices)

            x_train[[new_known, known_samples]] = x_train[[known_samples, new_known]]
            y_train[[new_known, known_samples]] = y_train[[known_samples, new_known]]
            known_samples += 1
    # Possible TODOs: Scaler, convert variables to categoricals (gender)


if __name__ == '__main__':
    main()
