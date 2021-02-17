import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import joblib

# Class starts from here
class CONVAI:
    #this is the empty vocabulary (vectorizer)
    cv = CountVectorizer(max_features = 20000) #change in no of features will result in how many different/unique words it will have
    classifier = GaussianNB() #this is the main algorith which works on probablistic approach
    no = 1000 #change this to change the number of data in terms of line you want to fed in model
    
    def init(self): #basic function 
        dataset = pd.read_csv('data.csv') #dataset loaded
        no=self.no
        corpus = [] #corpus will have cleaned data
        for i in range(0, no):
            review = re.sub('[^a-zA-Z]', ' ', dataset['0'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)
            
        print(corpus)
    
        
        X = self.cv.fit_transform(corpus).toarray() #divided dataset into 2 parts this will be like questions
        y = dataset.iloc[0:no, 2].values #this will be like answer to the abouve question
        # print(X)
        

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) #splitted dataset into train and test
        
        
        
        sav = self.classifier.fit(X_train, y_train) 
        
        y_pred = self.classifier.predict(X_test) #all the action is done here
        print(np.concatenate((y_pred.reshape(len(y_pred),1,), y_test.reshape(len(y_test),1)),1),) #printing the current actions
         

        cm = confusion_matrix(y_test, y_pred) 
        print(cm)
        a = accuracy_score(y_test, y_pred)
        print(a)
        joblib.dump(self.cv, "vectorizer1.pkl") #vocabulary is saved here
        joblib.dump(self.classifier, "classifier1.pkl") #algorithm is saved here


    # with open('model.pkl', 'wb') as fout:
    #     pickle.dump((cv, classifier), fout)

        # filename = 'finalized_model.sav'
        # pickle.dump(cv, open(filename, 'wb'))
        # filename = 'finalized.sav' 
        # pickle.dump(cv, open(filename, 'wb'))


    # saved_model = pickle.dumps(classifier)

    
    def Test(self,query): #this is the function for implementation of new inputs
        vectorizer = joblib.load("vectorizer.pkl") #vocabulary is loaded
        classifier = joblib.load("classifier.pkl") #algoritm is loaded

        # with open('model.pkl', 'rb') as fin:
        #     cv, classifier = pickle.load(fin)
        
        #This is known as preprocessing the data
        cv = self.cv
        classifier = self.classifier
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
        new_X_test = cv.transform(new_corpus).toarray() 
        new_y_pred = classifier.predict(new_X_test)
        print(new_y_pred)  #output from the algorithm is printed
        return new_y_pred  #output from the algorithm is returned
   
if __name__ == "__main__": #main class
    a=CONVAI() #created instance(object) of the class CONVAI
    a.init()  #called the function which will start training
    a.Test("hello") #enter different type of input here to get new output results   

