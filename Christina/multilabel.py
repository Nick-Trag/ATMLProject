import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import hamming_loss, zero_one_loss, multilabel_confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def svd_features(x_train):
    # dimensionality reduction using svd (appropriate for sparse matrices)
    svd = TruncatedSVD(n_components=50)
    x_train = svd.fit_transform(x_train)
    return x_train


def preprocess():
    fda = pd.read_csv("CAERS_ASCII_2004_2017Q2.csv")
    print("Data length ", len(fda))
    print("Percentage of null values: ", (fda.isnull().sum()/fda.shape[0])*100)
    print("Shape of Training Dataset:", fda.shape)

    # number of appearances of each value
    gender = fda['CI_Gender'].value_counts()
    print(gender)
    # drop not useful columns
    X = fda.drop(columns=['RA_Report #', 'RA_CAERS Created Date', 'AEC_Event Start Date', 'CI_Age at Adverse Event',
                          'CI_Age Unit', 'SYM_One Row Coded Symptoms'])
    # drop rows with missing values on Coded Symptoms
    X = X.dropna()
    # remove rows with not available values
    X = X[(X.CI_Gender == "Female") | (X.CI_Gender == "Male")]

    # get the target column and convert to lower case
    labels = X['AEC_One Row Outcomes']
    labels = labels.str.lower()

    X = X.drop(columns=['AEC_One Row Outcomes'])

    X = pd.get_dummies(X, prefix=['PRI_FDA Industry Name', 'PRI_Product Role', 'CI_Gender'],
                       columns=['PRI_FDA Industry Name', 'PRI_Product Role', 'CI_Gender'])

    # text process for the product names
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern='\w+')
    X['product'] = X.apply(lambda row: tokenizer.tokenize(row['PRI_Reported Brand/Product Name']), axis=1)
    # Convert the text to lower case
    X['product'] = X.apply(lambda row: [token.lower() for token in row['product']], axis=1)
    stopwords = ['while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    X['product'] = X.apply(lambda row: [token for token in row['product'] if token not in stopwords], axis=1)
    stemmer = PorterStemmer()
    X['product'] = X.apply(lambda row: [stemmer.stem(token) for token in row['product']], axis=1)
    X['product'] = X['product'].apply(' '.join)
    tfidf = TfidfVectorizer()
    tokens = tfidf.fit_transform(X['product'])
    svd = svd_features(tokens)
    X = X.join(pd.DataFrame(svd.tolist(),
                                    index=X.index))
    X = X.drop(columns=['PRI_Reported Brand/Product Name', 'product'])

    print("Shape of new Training Dataset:", X.shape)
    return X, labels


def binary_relevance(X, labels):

    x_train, x_test, y_train, y_test = transform_and_split(X, labels)

    # transformation using binary relevance
    classifier = BinaryRelevance(classifier=SVC(kernel='linear'), require_dense=[False, True])
    print("fitting...")
    classifier.fit(x_train, y_train)
    print("predicting...")
    y_pred = classifier.predict(x_test)
    print(y_pred[0])

    return y_pred, y_test


def label_powerset(X, labels):

    x_train, x_test, y_train, y_test = transform_and_split(X, labels)

    # transformation using label powerset
    classifier = LabelPowerset(classifier=DecisionTreeClassifier(random_state=0), require_dense=[False, True])
    print("fitting...")
    classifier.fit(x_train, y_train)
    print("predicting...")
    y_pred = classifier.predict(x_test)
    print(y_pred[0])

    return y_pred, y_test


def knn(X, labels):

    x_train, x_test, y_train, y_test = transform_and_split(X, labels)

    # use multi label knn for the prediction
    knn = MLkNN(k=10)
    print("fitting...")
    knn.fit(x_train, y_train)
    print("predicting...")
    y_pred = knn.predict(x_test)
    print(y_pred[0])

    return y_pred, y_test


def transform_and_split(X, labels):
    #use multi label binarizer to transform the labels
    mlb = MultiLabelBinarizer()
    print("Label example:")
    print(labels[0])
    labels = mlb.fit_transform(labels.str.split(','))
    print(labels[0])
    print(list(mlb.classes_))
    print('Number of unique labels: ', len(list(mlb.classes_)))

    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=100)
    return x_train, x_test, y_train, y_test


def evaluation(y_test, y_pred):

    print("Hamming loss: ", hamming_loss(y_test, y_pred))
    print("Zero one loss: ", zero_one_loss(y_test,y_pred, normalize=True))
    print('Accuracy ', str(accuracy_score(y_test, y_pred)))
    print('F1 score ', str(f1_score(y_test, y_pred, average='micro')))
    print('Precision ', str(precision_score(y_test, y_pred, average='micro')))
    print('Recall ', str(recall_score(y_test, y_pred, average='micro')))

    #print('Confusion matrix: ', multilabel_confusion_matrix(y_test, y_pred))
    label_names = [' congenital anomaly', ' death', ' disability', ' hospitalization', ' life threatening', ' non-serious injuries/ illness',
                   ' other serious (important medical events)', ' req. intervention to prvnt perm. imprmnt.', ' serious injuries/ illness', ' visited a health care provider',
                   ' visited an er', 'congenital anomaly', 'death', 'disability', 'hospitalization', 'life threatening', 'non-serious injuries/ illness', 'none', 'other serious (important medical events)',
                   'req. intervention to prvnt perm. imprmnt.', 'serious injuries/ illness', 'visited a health care provider', 'visited an er']

    print('Classification report: ', classification_report(y_test, y_pred, target_names=label_names))


X, labels = preprocess()
y_pred, y_test = binary_relevance(X, labels)
#y_pred, y_test = label_powerset(X, labels)
#y_pred, y_test = knn(X, labels)
evaluation(y_test, y_pred)
#label_based_macro_accuracy(y_test, y_pred)
#label_based_micro_accuracy(y_test, y_pred)
#label_based_macro_precision(y_test, y_pred)
