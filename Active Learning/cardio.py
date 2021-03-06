import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import entropy
from modAL.models import Committee, ActiveLearner
from modAL.disagreement import vote_entropy_sampling
from modAL.density import information_density

data_root = os.path.join(os.path.dirname(__file__), 'data')
RANDOM_STATE = 7
# np.random.seed(RANDOM_STATE)


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

    # 0.05% of the samples in the train set are actually labeled to start with
    initial_known_samples = int(0.0005 * len(x_train))
    known_samples = initial_known_samples
    max_iterations = 1000  # It's pointless to add every single sample to the labeled set, so we stop after 1000

    model = LogisticRegression()

    accuracies = np.zeros(min(len(y_train) - known_samples + 1, max_iterations))
    precisions = np.zeros_like(accuracies)
    recalls = np.zeros_like(accuracies)
    f1s = np.zeros_like(accuracies)

    for i in range(min(len(accuracies), max_iterations)):
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

            most_unsure = np.argmax(entropies) + known_samples  # (The indices of the probabilities and entropies arrays do not directly
            # correlate with the indices of x_train and y_train, so we need to add the number of known_samples)

            x_train[[most_unsure, known_samples]] = x_train[[known_samples, most_unsure]]
            y_train[[most_unsure, known_samples]] = y_train[[known_samples, most_unsure]]
            known_samples += 1

    random_accuracies = np.zeros_like(accuracies)
    random_precisions = np.zeros_like(accuracies)
    random_recalls = np.zeros_like(accuracies)
    random_f1s = np.zeros_like(accuracies)

    known_samples = initial_known_samples

    model = LogisticRegression()

    for i in range(min(len(random_accuracies), max_iterations)):
        print(i)
        model.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = model.predict(x_val)
        random_accuracies[i] = accuracy_score(y_val, y_pred)
        random_precisions[i] = precision_score(y_val, y_pred)
        random_recalls[i] = recall_score(y_val, y_pred)
        random_f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):
            new_known = np.random.randint(known_samples, len(x_train))

            x_train[[new_known, known_samples]] = x_train[[known_samples, new_known]]
            y_train[[new_known, known_samples]] = y_train[[known_samples, new_known]]
            known_samples += 1

    # QUERY BY COMMITTEE

    # The committee is comprised of 10 random shallow Decision Trees
    learners = [
        ActiveLearner(estimator=DecisionTreeClassifier(splitter='random', max_depth=1)) for __ in range(10)
    ]

    committee = Committee(learner_list=learners, query_strategy=vote_entropy_sampling)

    known_samples = initial_known_samples

    committee_accuracies = np.zeros_like(accuracies)
    committee_precisions = np.zeros_like(accuracies)
    committee_recalls = np.zeros_like(accuracies)
    committee_f1s = np.zeros_like(accuracies)

    for i in range(min(len(committee_accuracies), max_iterations)):
        print(i)
        committee.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = committee.predict(x_val)
        committee_accuracies[i] = accuracy_score(y_val, y_pred)
        committee_precisions[i] = precision_score(y_val, y_pred)
        committee_recalls[i] = recall_score(y_val, y_pred)
        committee_f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):
            most_unsure = committee.query(x_train[known_samples:])[0][0] + known_samples
            # committee.teach(x_train[most_unsure], y_train[most_unsure])

            x_train[[most_unsure, known_samples]] = x_train[[known_samples, most_unsure]]
            y_train[[most_unsure, known_samples]] = y_train[[known_samples, most_unsure]]
            known_samples += 1

    learners = [
        ActiveLearner(estimator=DecisionTreeClassifier(splitter='random', max_depth=1)) for __ in range(10)
    ]

    committee = Committee(learner_list=learners, query_strategy=vote_entropy_sampling)

    known_samples = initial_known_samples

    committee_random_accuracies = np.zeros_like(accuracies)
    committee_random_precisions = np.zeros_like(accuracies)
    committee_random_recalls = np.zeros_like(accuracies)
    committee_random_f1s = np.zeros_like(accuracies)

    for i in range(min(len(committee_random_accuracies), max_iterations)):
        print(i)
        committee.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = committee.predict(x_val)
        committee_random_accuracies[i] = accuracy_score(y_val, y_pred)
        committee_random_precisions[i] = precision_score(y_val, y_pred)
        committee_random_recalls[i] = recall_score(y_val, y_pred)
        committee_random_f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):
            new_known = np.random.randint(known_samples, len(x_train))

            x_train[[new_known, known_samples]] = x_train[[known_samples, new_known]]
            y_train[[new_known, known_samples]] = y_train[[known_samples, new_known]]
            known_samples += 1

    # plt.plot(accuracies, color='purple', label='Uncertainty Sampling')
    # plt.plot(random_accuracies, color='lightsalmon', label='Random Sampling')
    # plt.plot(committee_accuracies, color='chartreuse', label='Query By Committee')
    # plt.xlabel('Additional labeled examples')
    # plt.ylabel('Validation set accuracy')
    # plt.legend()
    # plt.show()

    plt.plot(np.convolve(accuracies, np.ones(7) / 7, mode='valid'), color='purple', label='Logistic Regression with Uncertainty Sampling')
    plt.plot(np.convolve(random_accuracies, np.ones(7) / 7, mode='valid'), color='lightsalmon', label='Logistic Regression with Random Sampling')
    plt.plot(np.convolve(committee_accuracies, np.ones(7) / 7, mode='valid'), color='chartreuse', label='Query By Committee')
    plt.plot(np.convolve(committee_random_accuracies, np.ones(7) / 7, mode='valid'), color='lightcoral', label='Committee with Random Sampling')
    plt.xlabel('Additional labeled examples')
    plt.ylabel('Validation set accuracy')
    plt.legend()
    plt.show()

    # DENSITY-WEIGHTED UNCERTAINTY SAMPLING

    # Only use 5000 samples from the training set, in order to make getting the density information feasible (the set has been shuffled, so the choice is actually random)
    train_samples_used = 5000
    x_train, y_train = x_train[:train_samples_used], y_train[:train_samples_used]

    known_samples = initial_known_samples

    density_accuracies = np.zeros(min(len(y_train) - known_samples + 1, max_iterations))
    density_precisions = np.zeros_like(density_accuracies)
    density_recalls = np.zeros_like(density_accuracies)
    density_f1s = np.zeros_like(density_accuracies)

    model = LogisticRegression()

    for i in range(min(len(density_accuracies), max_iterations)):
        print(i)
        # model.fit(x_train_l, y_train_l)
        model.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = model.predict(x_val)
        density_accuracies[i] = accuracy_score(y_val, y_pred)
        density_precisions[i] = precision_score(y_val, y_pred)
        density_recalls[i] = recall_score(y_val, y_pred)
        density_f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):

            probabilities = model.predict_proba(x_train[known_samples:])

            entropies = [entropy(probabilities[j], base=2) for j in range(len(probabilities))]

            densities = information_density(x_train[known_samples:], metric='euclidean')

            scores = entropies * densities

            most_unsure = np.argmax(scores) + known_samples

            x_train[[most_unsure, known_samples]] = x_train[[known_samples, most_unsure]]
            y_train[[most_unsure, known_samples]] = y_train[[known_samples, most_unsure]]
            known_samples += 1

    known_samples = initial_known_samples

    density_random_accuracies = np.zeros_like(density_accuracies)
    density_random_precisions = np.zeros_like(density_accuracies)
    density_random_recalls = np.zeros_like(density_accuracies)
    density_random_f1s = np.zeros_like(density_accuracies)

    model = LogisticRegression()

    for i in range(min(len(density_random_accuracies), max_iterations)):
        print(i)
        # model.fit(x_train_l, y_train_l)
        model.fit(x_train[:known_samples], y_train[:known_samples])
        y_pred = model.predict(x_val)
        density_random_accuracies[i] = accuracy_score(y_val, y_pred)
        density_random_precisions[i] = precision_score(y_val, y_pred)
        density_random_recalls[i] = recall_score(y_val, y_pred)
        density_random_f1s[i] = f1_score(y_val, y_pred)

        if known_samples != len(x_train):
            new_known = np.random.randint(known_samples, len(x_train))

            x_train[[new_known, known_samples]] = x_train[[known_samples, new_known]]
            y_train[[new_known, known_samples]] = y_train[[known_samples, new_known]]
            known_samples += 1

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(np.convolve(density_accuracies, np.ones(7) / 7, mode='valid'), color='crimson', label='Logistic Regression with Density-Weighted Uncertainty Sampling')
    plt.plot(np.convolve(density_random_accuracies, np.ones(7) / 7, mode='valid'), color='cornflowerblue', label='Logistic Regression with Random Sampling')
    plt.xlabel('Additional labeled examples')
    plt.ylabel('Validation set accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
