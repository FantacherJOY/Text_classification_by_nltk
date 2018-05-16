import os
import re
import string
import math
from decimal import Decimal
# use natural language toolkit
import pandas as pd
import nltk
from nltk.corpus import stopwords

from nltk.stem.lancaster import LancasterStemmer
stop_words = set(stopwords.words('english'))
# word stemmer
stemmer = LancasterStemmer()


import chardet
with open('data_for_spam.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

dataset=pd.read_csv('data_for_spam.csv', encoding=result['encoding'])

x=dataset.iloc[:,0]
y=dataset.iloc[:,1]
X=x.to_dict()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

s=y_train.reset_index()
y_train=s.iloc[:,1]

training_data = []
for d in range(len(X_train)):
    if y_train[d]=="spam":
        training_data.append({ "class":"spam","sentence":X_train[d]})
    else:
        training_data.append({ "class":"ham","sentence":X_train[d]})
    
print ("%s sentences of training data" % len(training_data))



#spam and ham count 

def prior_probabilty(y_train):
    spam_count=0
    ham_count=0
    for e in range(len(y_train)):
        if y_train[e]=="spam":
            spam_count=spam_count+1
        else:
            ham_count=ham_count+1
    return spam_count+ham_count


# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []
    
    
   
for data in training_data:
    
    # tokenize each sentence into words
    sen=data['sentence']
    #removing digit
    sentence= re.sub(r'\d+','', sen)
    sentence= re.sub('['+string.punctuation+']', '', sentence)
    #sentence=nltk.word_tokenize(sen.translate(dict.fromkeys(string.punctuation)))
    #for  nltk.work_tokenize(the_text.translate(dict.fromkeys(string.punctuation)))
    for word in nltk.word_tokenize(sentence):
        # removing digit puncuation
        if word not in stop_words:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])
            
#a=len(class_words['spam'])
#score+=(corpus_words[stemmer.stem(word.lower())]/prior_probabilty(y_train))
#Naive bayes classifier
d=1
def calculate_class_score_commonality(sentence, class_name, show_details=True):
    score=1
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        if word not in stop_words:
            if stemmer.stem(word.lower()) in class_words[class_name]:
                a=1
                a=a*(corpus_words[stemmer.stem(word.lower())])
                score=score*((a+1)/(len(class_words[class_name])+10))
#            score=log(score)
            #score=math.log10(score)
            #score/=corpus_words[stemmer.stem(word.lower())]
                if show_details:
                    print ("   match: %s" % stemmer.stem(word.lower() ))
            
            # treat each word with same weight
            
                
        # check to see if the stem of the word is in any of our classes
        
        
    return score
     

def classify(sentence):
    sentence= re.sub(r'\d+','', sentence)
    sentence= re.sub('['+string.punctuation+']', '', sentence)
    
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score_commonality(sentence, c, show_details=False)
        d=len(class_words[c])
     
        score=math.log(score)*math.log(1/d)
        if score > high_score:
            high_class = c
            high_score = score
            
    return high_class, high_score

z=[]
for j in range(len(X_test)):
    z.append(classify(X_test[j]))
#list to series
Z= pd.Series( (v[0] for v in z) )  
y_pred=Z
#confusion metrix  
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
cm = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)
Recall=recall_score(y_test, y_pred, average='weighted')
Precision=precision_score(y_test, y_pred, average='weighted')
