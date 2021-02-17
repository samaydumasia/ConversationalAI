# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 04:11:31 2021

@author: samay
"""
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import joblib

class Execute:
    
    def Test(self,query):
        vectorizer = joblib.load("vectorizer.pkl")
        classifier = joblib.load("classifier.pkl")
        print(vectorizer)
        # with open('model.pkl', 'rb') as fin:
        #     cv, classifier = pickle.load(fin)

        cv = vectorizer
        classifier = classifier
        #query = input()
        new_review = query
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower() 
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = vectorizer.transform(new_corpus).toarray()
        new_y_pred = classifier.predict(new_X_test)
        print(new_y_pred)
        return new_y_pred
    
if __name__ == "__main__":
    b = Execute()
    b.Test("what are you doing")    #enter different inputs here to test the model
    # vectorizer = joblib.load("vectorizer.pkl")
    # classifier = joblib.load("classifier.pkl")
    # print(classifier)
    # print(vectorizer)