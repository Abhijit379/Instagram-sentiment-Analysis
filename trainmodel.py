from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from load_data import load_train_test

def fit_logistic_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)  # Ensure convergence
    # Train model
    clf.fit(X_train, y_train)
    yy_test = clf.predict(X_test)
    print('Accuracy: {}'.format(accuracy_score(y_test, yy_test)))
    return clf

if __name__ == "__main__":
    print("Loading data")
    train_X, test_X, train_y, test_y = load_train_test()
    print("Extracting features (TF-IDF)")
    
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3))  # bigram features
    tfidf.fit(train_X)  # Fit on training data
    X_train_dtm = tfidf.transform(train_X)  # Transform training data
    X_test_dtm = tfidf.transform(test_X)  # Transform test data
    train_X, test_X = None, None  # Free memory

    print("Training logistic regression")
    current_clf = fit_logistic_regression(X_train_dtm, train_y, X_test_dtm, test_y)

    print("Saving artifacts")
    joblib.dump(dict(clf=current_clf, tfidf=tfidf), open("./cache/logistic.pkl", "wb"))
