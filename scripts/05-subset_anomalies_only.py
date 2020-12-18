"""
    Creates a subset of sentiment dataframe containing only anomalies
    Assigns an index to articles and their corresponding anomaly
"""

SENTIMENT_DATAFRAME = '../data_new/all_article_sentiments_v2_Reforma (Mexico).pkl'
ANOMALY_DATAFRAME = '../data_new/anomaly_date_df_7.pkl'


import pandas as pd
#from classifier import *
from tqdm import tqdm
tqdm.pandas()

anomaly_information = pd.read_pickle(ANOMALY_DATAFRAME)
cols = list(anomaly_information.columns.values)
anomaly_information.reset_index(inplace=True)
anomaly_information.columns = ['id'] + cols

#Load the pre-calculated sentiments
df_sentiments = pd.read_pickle(SENTIMENT_DATAFRAME) 


df_anomalies_only = pd.DataFrame()

def append_to_df(series):
    start = series['Start Date']
    end = series['End Date']
    agency = series['Agency']

    date_constraint = (df_sentiments['date'] >= start) & (df_sentiments['date'] <= end)
    agency_constraint = (df_sentiments[agency] == 1)
    
    #Select articles corresponding to an agency
    _tmp = df_sentiments[date_constraint & agency_constraint]
    #Indicate anomaly it belongs to
    _tmp['anomaly_id'] = series.id 
    return _tmp
    

for i in tqdm(range(anomaly_information.shape[0])):
    df_anomalies_only = pd.concat([df_anomalies_only, append_to_df(anomaly_information.iloc[i,:])])
    
df_anomalies_only.to_pickle('../data_new/sentiment_processed_anomalies_v4.pkl')