import os

import pandas as pd


data_root = './data'


def main():
    data = pd.read_csv(os.path.join(data_root, 'cardio', 'cardio_train.csv'), sep=';')


if __name__ == '__main__':
    main()
