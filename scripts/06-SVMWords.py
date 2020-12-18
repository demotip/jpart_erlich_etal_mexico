import random
import settings
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import pickle as pk
from tqdm import tqdm
tqdm.pandas()
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
stemmer = SnowballStemmer("spanish")
spanish_stopwords = set(stopwords.words('spanish')) 


"""
Gets the SVM words from the anomalies
"""


def getAnomalyData(agencies, start, end, article_df):
    '''
    Colect the anomaly articles and assign '1' to them
    '''
    anomaly_data=[]
    stop_punct = stopwords.words('spanish') + list(string.punctuation) + ['...','¿']
    for i, row in article_df.iterrows():

        if row.agency == agencies:
            if row.story is None:
                pass
            elif len(row.story) == 0:
                pass
            elif row.date >= start and row.date <= end:
                for articles in row.story:
                    clean_text = [word for word in word_tokenize(articles.lower()) if word not in stop_punct]
                    #clean_text = [word for word in articles.lower().split() if word not in stop_punct]
                    anomaly_data.append((' '.join(clean_text), 1))
                
    return anomaly_data

def getNonAnomalyData(agencies, start, end, article_df):
    
    '''   
    Collect the non-anomaly articles and assign '0' to them
    '''
    
    non_anomaly_data=[]
    stop_punct = stopwords.words('spanish') + list(string.punctuation) + ['...','¿']
    for i, row in article_df.iterrows():

        if row.agency == agencies:
            if row.date >= start and row.date <= end or row.story is None:
                pass
            elif len(row.story) > 0:
                for articles in row.story:
                    clean_text = [word for word in word_tokenize(articles.lower()) if word not in stop_punct]
                    non_anomaly_data.append((' '.join(clean_text), 0))
    return non_anomaly_data


def getVectorizedData(data):
    
    '''
    Vectorizes the data in order to run the SVM
    '''
    
    y = np.asarray([x[1] for x in data])
    x_data = [x[0] for x in data]
    count_vect_binary = CountVectorizer(binary = True, 
                                        ngram_range = (1,2), 
                                        max_df = 0.8, 
                                        min_df = 0, 
                                        max_features = 10000)
    x_vectors = count_vect_binary.fit_transform(x_data)    
    vocab = count_vect_binary.get_feature_names()
    
    return x_vectors, y, vocab


def svmWords(anomaly_df, article_df):
    
    '''
    Trains an SVM based on anomaly (1) and non-anomaly articles (0).
    Returns the most important features (words in this case) that distinguish an anomaly and a non-anomaly articles  
    '''
    
    svm_words = []
    svm_recall = []
    svm_precision = []
    for i, row in tqdm(anomaly_df.iterrows()):
        
        agencies = row['Agency']
        start = row['Start Date']
        end = row['End Date']

        non_anomaly_data = getNonAnomalyData(agencies,pd.to_datetime(start),pd.to_datetime(end),article_df)

        anomaly_data = getAnomalyData(agencies,pd.to_datetime(start),pd.to_datetime(end),article_df)

        data = anomaly_data+non_anomaly_data
        random.shuffle(data)

        X, y, vocab = getVectorizedData(data)
                
        avg_matrix = [0,0,0,0]
        coef_list = []
        
        # currently the number of splits is hard-coded at 3
        kf = KFold(n_splits=3)

        for train_index, test_index in kf.split(X):
            X_train = X[train_index]
            y_train = y[train_index]
            
            X_test  = X[test_index]
            y_test  = y[test_index]

            svc = SVC(kernel = 'linear')
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            
            coef_list.append(svc.coef_.toarray()[0])

            matrix = cm(y_test, y_pred, labels = [1, 0]).ravel()

            avg_matrix[0]=avg_matrix[0]+matrix[0]
            avg_matrix[1]=avg_matrix[1]+matrix[1]
            avg_matrix[2]=avg_matrix[2]+matrix[2]
            avg_matrix[3]=avg_matrix[3]+matrix[3]

        avg_matrix = [x/3 for x in avg_matrix]
        recall = avg_matrix[3]/(avg_matrix[2] + avg_matrix[3])
        precision = avg_matrix[3]/(avg_matrix[1] + avg_matrix[3])

        svm_recall.append(recall)
        svm_precision.append(precision)

        # avg coefficients from each split
        coef_sum = [sum(x) for x in zip(coef_list[0], coef_list[1], coef_list[2])]
        coef_avg = [x/3 for x in coef_sum]
        top_pos_coefs = np.argsort(coef_avg)[::-1][:20]
        pos_words = [vocab[i] for i in top_pos_coefs]
        svm_words.append(pos_words)

        print("{}/{}".format(i+1, len(anomaly_df)))
        
    return (svm_words, svm_recall, svm_precision)

####################################################


################################################################


ANOMALY_DATAFRAME = '../data_new/anomaly_date_df_7.pkl'
DAILY_DATAFRAME = '../data_new/daily_with_dict.pkl'

anomaly_information = pd.read_pickle(ANOMALY_DATAFRAME)
df_daily = pd.read_pickle(DAILY_DATAFRAME)
words, recall, precision = svmWords(anomaly_information, df_daily)

anomaly_information['SVM_words'] = pd.Series(words)

anomaly_information.to_pickle(settings.DATA_NEW+settings.ANOMALY_SVM)

"""
use this to load
with open('words_svm.pk','rb') as f:
    results = pk.load(f,encoding = 'latin1')
"""
