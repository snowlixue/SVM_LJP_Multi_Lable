# @Author : bamtercelboo
# @Datetime : 2019/6/10 13:00
# @File : aa.py
# @Last Modify Time : 2019/6/10 13:00
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  aa.py
    FUNCTION : None
"""


# from sklearn import datasets
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# print(X)
# print(y)
# svm = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
# result = svm.predict(X)
# print(result)


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])
y_train = [[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[0,1],[0,1]]
X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'hello welcome to new york. enjoy it here and london too'])
target_names = ['New York', 'London']

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    # ('clf', OneVsRestClassifier(LinearSVC()))])
    ('clf', OneVsRestClassifier(LinearSVC(kernel='linear',probability=True)))])
# SklearnClassifier(SVC(kernel='linear',probability=True))
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
for item, labels in zip(X_test, predicted):
    print('%s => %s' % (item, ', '.join(target_names[x] for x in labels)))

