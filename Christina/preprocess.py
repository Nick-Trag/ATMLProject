import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import hamming_loss, zero_one_loss


def svd_features(x_train):
    # dimensionality reduction using svd (appropriate for sparse matrices)
    svd = TruncatedSVD(n_components=100)
    x_train = svd.fit_transform(x_train)
    return x_train


def preprocess():
    fda = pd.read_csv("CAERS_ASCII_2004_2017Q2.csv")
    print(fda.head())
    print("Data length ", len(fda))
    print("Number of null values: ", fda.isnull().sum())
    print("Shape of Training Dataset:", fda.shape)

    X = fda.drop(columns=['RA_Report #', 'RA_CAERS Created Date', 'AEC_Event Start Date', 'CI_Age at Adverse Event',
                          'CI_Age Unit', 'AEC_One Row Outcomes'])

    X = X.dropna()
    # get the target column and convert to lower case
    labels = X['SYM_One Row Coded Symptoms']
    labels = labels.str.lower()
    print(labels[0])

    X = X.drop(columns=[ 'SYM_One Row Coded Symptoms'])
    X = pd.get_dummies(X, prefix=['PRI_FDA Industry Name', 'PRI_Product Role', 'CI_Gender'],
                       columns=['PRI_FDA Industry Name', 'PRI_Product Role', 'CI_Gender'])

    print("Shape of new Training Dataset:", X.shape)

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
    print(X.iloc[0])
    print(X.head())
    return X, labels


def problem_transform(X, labels):
    mlb = MultiLabelBinarizer()
    labels = labels.to_numpy()
    print("numpy labels")
    print(labels[0])
    labels = mlb.fit_transform(labels)
    print(labels[0])

    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=100)

    classifier = BinaryRelevance(classifier=DecisionTreeClassifier(random_state=0), require_dense=[False, True])
    print("fitting...")
    classifier.fit(x_train, y_train)
    print("predicting...")
    y_pred = classifier.predict(x_test)
    print(y_pred[0])

    return y_pred, y_test


def evaluation(y_test, y_pred):

    print("Hamming loss: ", hamming_loss(y_test, y_pred))
    print("Zero one loss: ", zero_one_loss(y_test,y_pred, normalize=True))
    print('Accuracy ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score ' + str(f1_score(y_test, y_pred, average='micro')))
    print('Precision ' + str(precision_score(y_test, y_pred, average='micro')))
    print('Recall ' + str(recall_score(y_test, y_pred, average='micro')))


X, labels = preprocess()
y_pred, y_tet = problem_transform(X, labels)
evaluation(y_tet,y_pred)
