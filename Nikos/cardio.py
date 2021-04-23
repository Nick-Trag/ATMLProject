import os

import pandas as pd
from sklearn.model_selection import train_test_split


data_root = os.path.join(os.path.dirname(__file__), 'data')
RANDOM_STATE = 7


def main():
    data = pd.read_csv(os.path.join(data_root, 'cardio', 'cardio_train.csv'), sep=';')
    x, y = data.drop(columns=['cardio']), data['cardio']
    # Split the data into train, validation and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    # Split the train set in 2. The unlabeled set, with 95% of samples and the labeled set, with only 5%.
    # From this point forward, y_train_u is considered unknown and will only be used as a replacement for a human annotator
    x_train_l, x_train_u, y_train_l, y_train_u = train_test_split(x_train, y_train, test_size=0.95, random_state=RANDOM_STATE)


if __name__ == '__main__':
    main()
