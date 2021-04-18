import pandas as pd
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier


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

    # get the target column and convert to lower case
    labels = fda['SYM_One Row Coded Symptoms']
    labels = labels.str.lower()
    print(labels[0])

    X = X.dropna()
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


preprocess()
