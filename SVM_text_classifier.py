# use natural language toolkit
import re
import numpy as np 
from random import randint
import pandas as pd
import string
import chardet
with open('data_for_spam.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

dataset=pd.read_csv('data_for_spam.csv', encoding=result['encoding'])
x=dataset.iloc[:,0]
y=dataset.iloc[:,1]
x=x.to_dict()

X=[]
for d in range(len(x)):
    b=x[d].lower()
    sentence= re.sub(r'\d+','', b)
    sentence= re.sub('['+string.punctuation+']', '', sentence)
    X.append(sentence)
   

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect=CountVectorizer()
a=count_vect.fit_transform(X)
a.toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.toarray()

from imblearn.over_sampling import SMOTE
sm=SMOTE()
X_train_res, y_train_res = sm.fit_sample(X_train_tfidf, y_train)

#unique, counts = np.unique(y_train_res, return_counts=True)

from sklearn.svm import SVC

clf= SVC(kernel = 'linear', random_state = 0)
clf.fit(X_train_res, y_train_res)
clf.score(X_train_res, y_train_res)


X_test_tfidf=count_vect.transform(X_test)

y_pred=clf.predict(X_test_tfidf)


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
cm = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)
Recall=recall_score(y_test, y_pred, average='weighted')
Precision=precision_score(y_test, y_pred, average='weighted')

#from sklearn.datasets import load_svmlight_files
#X_train, y_train, X_test, y_test = load_svmlight_files(['/path-to-file/train.txt', '/path-to-file/test.txt'])