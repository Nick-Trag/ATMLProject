import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import entropy


data_root = os.path.join(os.path.dirname(__file__), 'data')
RANDOM_STATE = 7


def main():
    data = pd.read_csv(os.path.join(data_root, 'cardio', 'cardio_train.csv'), sep=';')
    x, y = data.drop(columns=['cardio', 'id']), data['cardio']
    # Split the data into train, validation and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    # Split the train set in 2. The unlabeled set, with 95% of samples and the labeled set, with only 5%.
    # From this point forward, y_train_u is considered unknown and will only be used as a replacement for a human annotator
    x_train_l, x_train_u, y_train_l, y_train_u = train_test_split(x_train, y_train, test_size=0.95, random_state=RANDOM_STATE)

    model = LogisticRegression()
    model.fit(x_train_l, y_train_l)

    accuracies = np.zeros(len(y_train_u) + 1)
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    for i in range(len(accuracies)):
        y_pred = model.predict(x_val)
        accuracies[i] = accuracy_score(y_val, y_pred)
        precisions[i] = precision_score(y_val, y_pred)
        recalls[i] = recall_score(y_val, y_pred)
        f1s[i] = f1_score(y_val, y_pred)

        probabilities = model.predict_proba(x_train_u)

        entropies = [entropy(probabilities[j], base=2) for j in range(len(probabilities))]

        most_unsure = np.argmax(entropies)

    # Possible TODOs: Scaler, convert variables to categoricals (gender)


if __name__ == '__main__':
    main()
