"""
Get the top articles based on the SVM words
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

def buildCorruptionDict2(text_list, wordlist):
    '''
    takes either anomaly_df.all_text or anomaly_df.matching_text and a word list
    for each anomaly, build the corruption-word dict, return list of dicts

    
    '''
    
    
    dict_list = []
    for i, text_str in enumerate(text_list):
        term_dict = {}
        if text_str is None:
            dict_list.append(None)
        else:
            for word in wordlist:

                term_dict[word]=text_str.count(word)


            dict_list.append(term_dict)
        
        
    return dict_list

def fix_sentences(df_fix):
    '''
    takes in standard article df, copies and separates rows with multiple entities
    returns new article df with separated rows
    
    '''
    df_toclean = df_fix.copy()
    agency = df_toclean.agencies.values
    story_sentence = df_toclean.story_sentence_index.values
    for i in range(len(story_sentence)):
        separate_agencies = story_sentence[i][agency[i]]
        story_sentence[i] = separate_agencies
            
    
    
    return(df_toclean)


def sepMultiEntityRows(article_df):
    '''
    takes in standard article df, copies and separates rows with multiple entities
    returns new article df with separated rows
    
    '''
    separate_stories = []
    for i, row in article_df.iterrows():
        if len(row.agencies) > 1:
            for agency in row.agencies:
                cpy_row = row.copy(deep=True)
                cpy_row.agencies = agency # change agencies from list to just name of this agency
                cpy_row.senti_avg_per_agency  = row.senti_avg_per_agency[agency]# set this to just the score for the cur agency
                cpy_row.buffered_story_sentence_index = row.buffered_story_sentence_index[agency]
                separate_stories.append(cpy_row)
        
        else:
            update_row = row.copy(deep=True)
            update_row.agencies = row.agencies[0] # turns the list into just the first entry 
            update_row.senti_avg_per_agency = row.senti_avg_per_agency[update_row.agencies] # turns the dict into just the dict value
            update_row.buffered_story_sentence_index = row.buffered_story_sentence_index[row.agencies[0]]
            separate_stories.append(update_row)
        
        
    sep_df = pd.DataFrame(data = separate_stories)
    
    return sep_df


def buildDF(article_df):
    
    
    sep_df = sepMultiEntityRows(article_df)
    sep_df = fix_sentences(sep_df)
    
    return sep_df


def subset_anomalies(df_daily,anomaly_information):
    cols = list(anomaly_information.columns.values)
    anomaly_information.reset_index(inplace=True)
    anomaly_information.columns = ['anomaly_id'] + cols




    df_anomalies_only = pd.DataFrame()

    def append_to_df(series):
        start = series['Start Date']
        end = series['End Date']
        agency = series['Agency']

        date_constraint = (df_daily['date'] >= start) & (df_daily['date'] <= end)
        agency_constraint = df_daily['agencies'] == agency
        all_constraints = date_constraint  & agency_constraint

        #Select articles corresponding to an agency
        _tmp = df_daily[all_constraints.values]
        #Indicate anomaly it belongs to
        _tmp['anomaly_id'] = series.anomaly_id
        return _tmp


    for i in tqdm(range(anomaly_information.shape[0])):
        df_anomalies_only = pd.concat([df_anomalies_only, append_to_df(anomaly_information.iloc[i,:])])

    return df_anomalies_only

def get_top_articles(df,anomaly_information,svm_words):
    num_anomalies = len(svm_words)
    
    for anomaly in range(num_anomalies):
        
        df_anom = df[df['anomaly_id']==anomaly]
        words = svm_words[anomaly]
        svm_dict = buildCorruptionDict2(df_anom.story, words)
        words_count = [sum(d.values()) for d in svm_dict]
        top_10 = np.flip(np.argsort(words_count))
        file = open("top_articles_text_new/Anomaly_"+str(anomaly)+".txt", "w")
        header_file = 'Agency '+str(anomaly_information.loc[anomaly,'Agency'])+'\nSVM Words \n' + ''.join(words) + '\n'
        dates_info = '\nStart Date: '+ str(anomaly_information.loc[anomaly,'Start Date']) + ' End Date: '+ str(anomaly_information.loc[anomaly,'End Date'])   
        file.write(header_file)
        file.write(dates_info + '\n \n')
        for k,idx in enumerate(top_10):
            header_anomaly = f'\n \n \n========== TOP {k+1} STORY ========== \n \n \n'  
            text = df_anom.loc[df_anom.index[idx],'story']
            id_info = 'Article ID: ' + str(df_anom.loc[df_anom.index[idx],'id']) + '\nArticle Date: '+str(df_anom.loc[df_anom.index[idx],'date'])
            story_info = '\nStory title: '+ str(df_anom.loc[df_anom.index[idx],'title']) + '\n' + str(df_anom.loc[df_anom.index[idx],'section'])
            file.write(header_anomaly)
            file.write(id_info + '\n')
            file.write(story_info+ '\n')
            file.write('\n\nStory\n\n'+text+ '\n \n \n \n') 
        
        file.close() 
            

df_svm = pd.read_csv('../../data_new/aggregated_anomaly_svm_stop_max10_alpha10.csv')
df_all = pd.read_pickle('../../data_new/all_article_sentiments.pkl')[pd.read_pickle('../../data_new/all_article_sentiments.pkl')['source'] == 'Reforma (Mexico)']
anomaly_information = pd.read_pickle('../../data_new/anomaly_date_df_100.1.pkl')

df = buildDF(df_all)
df = subset_anomalies(df.loc[:,['id','title','section','date','story','agencies']],anomaly_information)

get_top_articles(df,anomaly_information,df_svm['SVM_words'])
