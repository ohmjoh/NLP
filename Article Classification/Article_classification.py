#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 01:31:28 2018

@author: minoh
"""

#%% Import libraries

import nltk
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from statistics import mean
from statistics import stdev
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn


#%% Import and prepare data
topic = pd.read_csv('topic.csv')
topic = topic.drop(columns=['url'])

# Combine title and body
topic['titlebody'] = topic['title'] + ' ' + topic['body']

# Find duplicates
for i, texti in enumerate(topic['titlebody']):
    for j, textj in enumerate(topic['titlebody']):
        if texti == textj and i!=j:
            print(i) #find duplicates
    #index 126 matches index 129, index 1300 matches index 1301, 
    #index 1459 matches index 1479, index 1536 matches index 1537
    
# Delete duplicates
topic = topic.drop(topic.index[1537])
topic = topic.drop(topic.index[1479])
topic = topic.drop(topic.index[1301])
topic = topic.drop(topic.index[129])

# Data exploration
topic.describe()
topic.info()
topic.groupby('annotation').describe()

# Plot histogram
fig = plt.figure(figsize=(8,6))
topic.groupby('annotation').annotation.count().plot.bar(ylim=0)
plt.show() #data is imbalanced

# Shuffle data
topic_shuffled = shuffle(topic)
topic_shuffled = topic_shuffled.sample(frac=1).reset_index(drop=True)

# Preparation for 10-fold cross validation
# (Edited from https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)
kf = KFold(n_splits=10)
X = topic_shuffled['titlebody']
X_title = topic_shuffled['title']
X_body = topic_shuffled['body']
y = topic_shuffled['annotation']
kf.get_n_splits(X) #output: 10

# =============================================================================
#%% Baseline

def base_tokenize(text):
    tokens = nltk.word_tokenize(text)
    return(tokens)  

def experiment0(X, y):
    """
    Baseline implementation using bag-of-words (unigram) approach.
    
    Args: 
        X (pandas Series): list of text data
        y (pandas Series): list of labels
    
    Returns: 
        errors (pandas DataFrame): list of missclassifed texts with actual and
        predicted classes
            
    This function does following steps:
    1. Produce a sparse representation of the token counts
    2. Fit logistic regression model to the data
    3. Make predictions using 10-fold cross validation
    4. Calculate and print metrics
    5. Return a misclassification list for error analysis
    """
    accuracy_list = []
    sum_feat_num = 0
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    precision_list = []
    recall_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = CountVectorizer(tokenizer=base_tokenize, lowercase=False).fit(X_train)
        
        feat_num = len(vect.get_feature_names())
        print(test_index)
        
        # bag of wards representation
        X_train_vectorised = vect.transform(X_train)
        
        # Most frequent features
        # (Edited from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d)
        sum_words = X_train_vectorised.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        print('30 most frequent features:')
        print(words_freq[:30])
        #print('30 least frequent features:')
        #print(words_freq[-30:])
        
        # Logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Compute and print metrics
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data
        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[:-11:-1] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))
            #print(top10)
            
            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  " ".join(feature_names[j] for j in bottom10)))
            
            
        # Binary classification (virality data) - Uncomment to run with virality data
        #feature_names = np.array(vect.get_feature_names())
        #sorted_coef_index = model.coef_[0].argsort()
        #print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10])) #low virality
        #print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1])) #high virality

        
        # Misclassification list 
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
    
    # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
    
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)
    return errors

#Uncomment to run 
#exp0_errors = experiment0(X,y)
    
# =============================================================================
#%% Concordance   
def get_concordance(word, textlist):
    """
    Print out the concordance of a word in a list of text
    """
    for text in textlist:
        tokens = nltk.word_tokenize(text)
        ci = nltk.ConcordanceIndex(tokens)
        if ci.offsets(word):
            ci.print_concordance(word)
            
# Example usage
# get_concordance("U.S", X)
# Displaying 1 of 1 matches:
#  have invited the pope to visit the U.S Capitol and address a joint session
# Displaying 1 of 1 matches:
# mers alike . -- Bill Vlasic * Major U.S Banks to Report Earnings The countr
# Displaying 1 of 1 matches:
#  HAS PUBLICLY CALLED FOR ATTACKS ON U.S . TARGETS OVERSEAS . AND HERE AT HO
#  ....
 
# =============================================================================
#%%
def whereis(word, error_list):
    """
    Print out the concordance of a word in a error (misclassification) list returned
    from the experiment# functions. 
    Index, actual class, predicted class and conconrdance of the word get printed.
    error_list 
    """
    for i, row in error_list.iterrows():
        if word in error_list.loc[i]['Text']:
            print(i)
            print(error_list.loc[i]['Actual'])
            print(error_list.loc[i]['Prediction'])  
            get_concordance(word, [error_list.loc[i]['Text']])

# Example usage           
# whereis('@', exp0_errors)
# 33
# Society
# Science and Technology
# Displaying 1 of 1 matches:
# the full report . Follow LiveScience @ livescience , Facebook & Google+ .
# 66
# Politics
# Society
# Displaying 1 of 1 matches:
# to us via the form below or tweet it @ YourMirror with the hashtag # celebr
# ....
            
            
            
# =============================================================================       
#%% Experiment 1: Cleaning, abbreviation treatment
def preprocess(text_list):
    """
    Take a text_list (pandas.Series) as an argument and 
    return a cleaned text list.
    """
    clean = []
    for text in text_list:
        t = re.sub("http:\/\/[\w\.]+", "", text) #remove web addresses
        t = re.sub("[\w.]*@[\w.]+", "", t) #remove emails or twitter accounts
        t = re.sub("USA TODAY", "", t) #remove publication to avoid confusion with the country name, US
        t = re.sub("( U\.?S ?\.?)|(United States)" , " usa ", t) #standardise the abbreviated term for the US to usa
        t = re.sub("(ISIL)|(Isil)|(Islamic State)", " isis ", t) #standardise the abbreviated term for the ISIS to isis  
        t = re.sub("\n", " ", t) #remove newline character
        t = re.sub("'s\b", " ", t) #remove 's
        t = re.sub("\.", "", t) #make other abbreviation words without the full stop
        t = re.sub("[0-9]+", "NUM", t.lower()) #replace numbers
        t = re.sub("[^a-zA-Z]", " ", t) #remove non-alphbetic characters
        clean.append(t)
        clean_text = pd.Series(clean)
    return clean_text

#Uncomment to run
#X_clean = preprocess(X)
    

def experiment1(X,y):
    """
    Run experiment0 with the cleaned data.
    Must use cleaned data for the first argument X.
    """
    errors = experiment0(X,y)
    return errors

#Uncomment to run
#exp1_errors = experiment1(X_clean, y)

# =============================================================================
#%% Experiment 2: Bigram
def experiment2(X, y):
    """
    Use unigrams and bigrams for the features to train the model. 
    X: cleaned text data
    y: label
    
    Return a misclassification list
    
    """
    accuracy_list = []
    sum_feat_num = 0
    precision_list = []
    recall_list = []
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = CountVectorizer(tokenizer=base_tokenize, lowercase=False, ngram_range=(1,2)).fit(X_train)
        feat_num = len(vect.get_feature_names())
        #print('Number of features:')
        #print(feat_num)
        #print(test_index)
        
        #bag of wards representation
        X_train_vectorised = vect.transform(X_train)
        
        # Most frequent features
        # Edited from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
        sum_words = X_train_vectorised.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        print('30 most frequent features:')
        print(words_freq[:30])
        #print('30 least frequent features:')
        #print(words_freq[-30:])
        
        # Logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Compute and print metrics   
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data
        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[:-11:-1] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))

            #print(top10)
            
            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  " ".join(feature_names[j] for j in bottom10)))
            
        # Binary classification (virality data) - Uncomment to run with virality data
        #feature_names = np.array(vect.get_feature_names())
        #sorted_coef_index = model.coef_[0].argsort()
        #print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10])) #low virality
        #print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1])) #high virality

        
        # Misclassification list 
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
    
    # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
    
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)
    return errors
 
#Uncomment to run
#exp2_errors = experiment2(X_clean, y)
    


# =============================================================================
#%% Experiment 3: Stopwords, TF-IDF
def stopwords_tokenize(text):
    tokens = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in tokens:
        if w not in stopWords:
            wordsFiltered.append(w)
    return(wordsFiltered)
    
def experiment3_1(X,y):
    """ 
    Remove stopwords through the tokenizer (stopwords_tokenize()). 
    Use unigrams and bigrams.
    Return the misclassification list.
    
    """
    accuracy_list = []
    sum_feat_num = 0
    precision_list = []
    recall_list = []
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = CountVectorizer(tokenizer=stopwords_tokenize, lowercase=False, ngram_range=(1,2)).fit(X_train)
        feat_num = len(vect.get_feature_names())
        #print('Number of features:')
        #print(feat_num)
        #print(test_index)
        
        #bag of wards representation
        X_train_vectorised = vect.transform(X_train)
        
        # Most frequent features
        # Edited from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
        sum_words = X_train_vectorised.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        print('30 most frequent features:')
        print(words_freq[:30])
        #print('30 least frequent features:')
        #print(words_freq[-30:])
        
        # Logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Compute and print metrics   
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data
        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[:-11:-1] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))

            #print(top10)
            
            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  " ".join(feature_names[j] for j in bottom10)))
            
            
        # Binary classification (virality data) - Uncomment to run with virality data
        #feature_names = np.array(vect.get_feature_names())
        #sorted_coef_index = model.coef_[0].argsort()
        #print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10])) #low virality
        #print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1])) #high virality

        
        # Misclassification list 
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
    
    # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
    
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)
    return errors  

#exp3_1_errors = experiment3_1(X_clean, y)

def experiment3_2(X, y):
    """ 
    Remove stopwords through the tokenizer (stopwords_tokenize()). 
    Use unigrams and bigrams.
    Use TF-IDF.
    Return the misclassification list.
    
    """
    accuracy_list = []
    sum_feat_num = 0
    precision_list = []
    recall_list = []
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = TfidfVectorizer(tokenizer=stopwords_tokenize, lowercase=False, ngram_range=(1,2)).fit(X_train)
        feat_num = len(vect.get_feature_names())
        #print('Number of features:')
        #print(feat_num)
    
        #bag of words representation
        X_train_vectorised = vect.transform(X_train)
        
        #logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Metrics   
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data
        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[-10:] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))

            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  " ".join(feature_names[j] for j in bottom10)))
            
        # Binary classification (virality data) - Uncomment to run with virality data
        #feature_names = np.array(vect.get_feature_names())
        #sorted_coef_index = model.coef_[0].argsort()
        #print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10]))
        #print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1]))
        #Largest, smallest tf-idf
        #sorted_tfidf_index = X_train_vectorised.max(0).toarray()[0].argsort()
        #print('Smallest tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
        #print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
        
        # Misclassification list
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
                 
              
    # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
    
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)
    return errors  

#Uncomment to run
#exp3_1_errors = experiment3_1(X_clean, y)  
#exp3_2_errors = experiment3_2(X_clean, y)

# =============================================================================
#%% Experiment 4: Stemming

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = [stemmer.stem(item) for item in tokens]
    return(stemmed)    
    
def stem_tokenize(text):
    tokens = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in tokens:
        if w not in stopWords:
            wordsFiltered.append(w)
    stems = stem_tokens(wordsFiltered, stemmer)
    return(stems)
    
def experiment4(X, y):
    """ 
    Remove stopwords and stem through the tokenizer (stem_tokenize()). 
    Use unigrams and bigrams.
    Return the misclassification list.
    
    """
    accuracy_list = []
    sum_feat_num = 0
    precision_list = []
    recall_list = []
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = CountVectorizer(tokenizer=stem_tokenize, lowercase=False, ngram_range=(1,2)).fit(X_train)
        feat_num = len(vect.get_feature_names())
        #print('Number of features:')
        #print(feat_num)
        #print(test_index)
        
        #bag of wards representation
        X_train_vectorised = vect.transform(X_train)
        
        # Most frequent features
        # Edited from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
        sum_words = X_train_vectorised.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        print('30 most frequent features:')
        print(words_freq[:30])
        #print('30 least frequent features:')
        #print(words_freq[-30:])
        
        # Logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Metrics   
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data
        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[:-11:-1] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))
            
            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  " ".join(feature_names[j] for j in bottom10)))
            
        # Binary classification (virality data) - Uncomment to run with virality data
        #feature_names = np.array(vect.get_feature_names())
        #sorted_coef_index = model.coef_[0].argsort()
        #print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10])) #low virality
        #print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1])) #high virality

        
        # Misclassification list 
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
    

     # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
     
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)

    return errors

#Uncomment to run
#exp4_errors = experiment4(X_clean, y)  


# =============================================================================
#%% Experiment 5: Wordnet synsets
def wordnet_synset(tokens):
    syns = []
    for item in tokens:
        if len(wn.synsets(item)) > 1:
            syns.append(wn.synsets(item)[1].name())
            
    for item in tokens:
        syns.append(item)
    return(syns)    

def wordnet_hypernym(tokens):
    hypers = []
    for item in tokens:
        syns =  wn.synsets(item)
        if len(syns) > 0:
            hyper = syns[0].hypernyms()
            if len(hyper) > 0:
                hypers.append(hyper[0].name())
    for item in tokens:
        hypers.append(item)
    return(hypers)
    
    
def syns_tokenize(text):
    tokens = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in tokens:
        if w not in stopWords:
            wordsFiltered.append(w)
    syns = wordnet_synset(wordsFiltered)
    return(syns)
    
def hypers_tokenize(text):
    tokens = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in tokens:
        if w not in stopWords:
            wordsFiltered.append(w)
    hypers = wordnet_hypernym(wordsFiltered)
    return(hypers)
    
    
def experiment5_1(X, y):
    """ 
    Remove stopwords and add synsets of words through tokenizer (syns_tokenize). 
    Use unigrams and bigrams.
    Return the misclassification list.
    
    """
    accuracy_list = []
    sum_feat_num = 0
    precision_list = []
    recall_list = []
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = CountVectorizer(tokenizer= syns_tokenize, ngram_range=(1,2)).fit(X_train)
        feat_num = len(vect.get_feature_names())
        #print('Number of features:')
        #print(feat_num)
        #print(test_index)
        
        #bag of wards representation
        X_train_vectorised = vect.transform(X_train)
        
        # Most frequent features
        # Edited from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
        sum_words = X_train_vectorised.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        print('30 most frequent features:')
        print(words_freq[:30])
        #print('30 least frequent features:')
        #print(words_freq[-30:])
        
        # Logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Metrics   
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data
        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[:-11:-1] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))

            
            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  " ".join(feature_names[j] for j in bottom10)))
            
        # Binary classification (virality data) - Uncomment to run with virality data
        #feature_names = np.array(vect.get_feature_names())
        #sorted_coef_index = model.coef_[0].argsort()
        #print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10])) #low virality
        #print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1])) #high virality

        
        # Misclassification list 
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
    
    # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
    
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)
    return errors

def experiment5_2(X, y):
    """ 
    Remove stopwords and add hypernyms of words through tokenizer (syns_tokenize). 
    Use unigrams and bigrams.
    Return the misclassification list.
    
    """
    accuracy_list = []
    sum_feat_num = 0
    precision_list = []
    recall_list = []
    errors = pd.DataFrame(columns = ['Actual', 'Prediction', 'Text'])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        vect = CountVectorizer(tokenizer= hypers_tokenize, ngram_range=(1,2)).fit(X_train)
        feat_num = len(vect.get_feature_names())
        
        #bag of wards representation
        X_train_vectorised = vect.transform(X_train)
        
        # Most frequent features
        # (Edited from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d)
        sum_words = X_train_vectorised.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        print('30 most frequent features:')
        print(words_freq[:30])
        #print('30 least frequent features:')
        #print(words_freq[-30:])
        
        # Logistic regression
        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        predictions = model.predict(vect.transform(X_test))
        
        # Compute and print metrics   
        # Accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        print('Accuray:')
        print(accuracy)
        # Confusion Matrix
        #df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
        #print(df_confusion)
        # Precision, recall, f-score
        print('Precision, recall, f-score:')
        print (classification_report(y_test, predictions))
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        precision_list.append(precision)
        recall_list.append(recall)
        sum_feat_num += feat_num
        
        # Print features with the highest coefficient values per class
        
        # Mulitclass - Comment out when running with virality data

        feature_names = vect.get_feature_names()
        class_labels = ['Business', 'Entertainment', 'Error', 'Health', 'Other', 'Politics', 'Science and Technology', 
                        'Society', 'Sports', 'War']
        for i, class_label in enumerate(class_labels):
            top10 = np.argsort(model.coef_[i])[:-11:-1] # Most informative
            print("%s: %s" % (class_label,
                              ", ".join(feature_names[j] for j in top10)))

            
            #bottom10 = np.argsort(abs(model.coef_[i]))[:10] # Most irrelevant. Bottom 10 of the absolute values of coefficients 
            #print("%s: %s" % (class_label,
            #                  ", ".join(feature_names[j] for j in bottom10)))
            
        # Binary classification (virality data) - Uncomment to run with virality data
        feature_names = np.array(vect.get_feature_names())
        sorted_coef_index = model.coef_[0].argsort()
        print('Smallest Coef:\n{}\n '.format(feature_names[sorted_coef_index][:10])) #low virality
        print('Largest Coef:\n{}'.format(feature_names[sorted_coef_index][:-11:-1])) #high virality

        
        # Misclassification list 
        error_idx = len(errors)
        for i, (actual, text) in enumerate(zip(y_test, X_test)):
            if actual != predictions[i]:
                errors.loc[error_idx + i] = [actual, predictions[i], text]
    
    # Plot P-R          
    fig = plt.figure(figsize=(5,5))    
    plt.plot(recall_list, precision_list, 'ro', markeredgecolor='k')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()   
    
    avg_acc = mean(accuracy_list)
    sd_acc = stdev(accuracy_list)
    avg_feat_num = sum_feat_num/10
    print('Avg Feature Number:')
    print(avg_feat_num)
    print('Avg Accuracy:')
    print(avg_acc)
    print('Accuracy Standard Deviation:')
    print(sd_acc)
    return errors

#Uncomment to run
#exp5_1_errors = experiment5_1(X_clean, y) 
#exp5_2_errors = experiment5_2(X_clean, y) 
    

# =============================================================================
#%% Virality
# Import and prepare data
virality = pd.read_csv('virality.csv')
virality.head()
virality = virality.drop(columns=['url'])


# Combine title and body
virality['titlebody'] = virality['title'] + ' ' + virality['body']

# Find duplicates
for i, texti in enumerate(virality['titlebody']):
    for j, textj in enumerate(virality['titlebody']):
        if texti == textj and i!=j:
            print(i) #find duplicates
    

# Delete duplicates
virality = virality.drop(topic.index[1537])
virality = virality.drop(topic.index[1479])
virality = virality.drop(topic.index[1301])
virality = virality.drop(topic.index[129])

# Data exploration
virality.describe()
virality.info()
virality.groupby('class').describe() #high 1230, low 1051


# Shuffle data
virality_shuffled = shuffle(virality)
virality_shuffled = virality_shuffled.sample(frac=1).reset_index(drop=True)

# Preparation for 10-fold cross validation
# (https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)
kf = KFold(n_splits=10)
X_viral = virality_shuffled['titlebody']
y_viral = virality_shuffled['class']
kf.get_n_splits(X_viral) #output: 10

# Run experiments for virality data
exp0_v = experiment0(X_viral, y_viral)
X_viral_clean = preprocess(X_viral)
exp1_v = experiment1(X_viral_clean, y_viral)
exp2_v = experiment2(X_viral_clean, y_viral)
exp3_1_v = experiment3_1(X_viral_clean, y_viral)
exp3_2_v = experiment3_2(X_viral_clean, y_viral)
exp4_v = experiment4(X_viral_clean, y_viral)
exp5_1_v = experiment5_1(X_viral_clean, y_viral)
exp5_2_v = experiment5_2(X_viral_clean, y_viral)




