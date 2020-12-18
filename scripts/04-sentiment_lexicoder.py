
"""
Script to get the sentiment analysis based on the spanish lexicoder.
It calls a R script defined by variable r_file
This script receives a processed dataframe with the text of both full story and buffer.
It returns a dataframe with the sentiment scores that is then merged by this script.
"""

import pandas as pd
from tqdm import tqdm
#import settings
import utils
import numpy as np
tqdm.pandas()
import re
import spacy
import subprocess

### Defining preprocessing parameters

# spacy pipeline
nlp = spacy.load('es_core_news_md')

# tags to remove (see https://spacy.io/api/annotation )
tags = ["PRON","DET","ADP","PUNCT","CONJ","CCONJ","SCONJ","NUM","SYM","SPACE"]

# the tags above should filter pronouns, but we add a custom list just in case
es_pron = ["yo","tú","él","ella","nosotros","ustedes","vosotros","ellos","ellas", 
           "te","me","lo","la","les","nos","vos","os","le"]





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




def appendBuffer(article_df):
    '''
    takes in standard article df, copies and separates rows with multiple entities
    returns new article df with separated rows
    
    '''
    separate_stories = []
    for i, row in article_df.iterrows():
        cpy_row = row.copy(deep=True)
        agency = row.agencies
        cpy_row.story  = row.story
        
        out_list = []
        to_buffer = []
        for idx in row.story_sentence_index[agency]:
            start = max(0,idx-1)
            end = min(row.num_sentences,idx+2)
            text = np.array(row.story_sentences)[list(range(start,end))].tolist()
            to_buffer+=text

        out_list.append(int(row.id))
        out_list.append(agency)
        out_list.append(row.story)
        out_list.append(' '.join(to_buffer))
        
        separate_stories.append(out_list)

        
    sep_df = pd.DataFrame(data = separate_stories,columns = ['Article_ID','Agency','Story','Buffer_Text'])
    
    return sep_df



            
# function that will lemmatize the texts
def lemmatize_text(text, tags = tags, 
                   lang='es', 
                   string=True, 
                   filter_stopwords=False, 
                   prons_list = es_pron, 
                   recursive=False,
                   recursive_out=True): 
    """ Arguments: 
            @ text : the input text 
            @ tags : spaCy tags to filter
            @ lang : input text's language
            @ string : logical. If True, will only returns lemmas
            @ filter_stopwords : logical
            @ prons_list : custom pronouns list to filter
            @ recursive : will assume input is a list of texts. 
            @ recursive_out: if True, will ouput a list of lists
         Returns: 
             @ list of (token, lemma) tuples if string=False
             @ list of lemmas if string=True
        """
    stop_w = []
    
    # Assign stop words
    if filter_stopwords == True:
        from stop_words import get_stop_words
        stop_w = get_stop_words(lang) # obtain according to lang
        
    lemmas = []
        
    if recursive:  
        # process each doc in the input list
        for doc in nlp.pipe(text): 
            # filter token tags, numeric, stopwords (if any)
            
            temp = [token.lemma_ for token in doc if token.pos_ not in tags 
              and token.text.isalpha() and token.text not in stop_w     
              and token.text not in prons_list and len( str(token.text)) >= 1]
            if recursive_out: 
                lemmas += [[' '.join(temp)]]
            else: 
                lemmas += [' '.join(temp)]
            
        return lemmas
        
    else: 
        # process a collection of texts -> pass a pipeline
        doc = nlp(text) 
            
        # Filter lemmas by tags, alphanumeric, and more. 
        lemmas = [(token.text, token.lemma_) for token in doc if token.pos_ not in tags 
                  and token.text.isalpha() and token.text not in stop_w     
                  and token.text not in prons_list and len( str(token.text)) >= 1]
    
    
    # if strrng, only care bout the actual lemmas. 
    if string: 
        lemmas = [ tup[1] for tup in lemmas] # (token.text, token.lemma_)
        return ' '.join(lemmas) # Return lemmas 
    else: 
        return lemmas
    

def clean_up(bad_text, recursive=True): 
    """ Filters unwanted character in the input text. 
    Arguments:
    @ recursive: If True, return a list splitted by comma. 
    """
    temp = re.sub(r'[\[\]\"\'\n\t\0]|^[\s\t\0]*|\s$', '', bad_text)
    if recursive: 
        temp = re.split(',', temp)
        return temp
#    temp = [w for w in temp if len(w) > 1]
    return temp

################ Preparing the article stories and buffers ################

df = pd.read_pickle('../data_new/all_article_sentiments.pkl')

df_separated = sepMultiEntityRows(df)

df_buffer = appendBuffer(df_separated)


################ Lemmatizing full text and buffer ################

df_buffer['lemmatized_full_text'] = df_buffer['Story'].apply(lemmatize_text, prons_list=es_pron, tags=tags)
df_buffer['lemmatized_buffer'] = df_buffer['Buffer_Text'].apply(lemmatize_text, prons_list=es_pron, tags=tags)

################ parser arguments ################

input_name = 'agency_news_with_buff_mod.csv' # input file
column_num = '0' # column to perform SA, 0 buffer, 1 full story
cut = '0'  # number of cuts
output_name = 'metrics_agency'  # name of the output file

r_file = 'Agency_newspaper_metrics.r'


df_buffer.to_csv(input_name)

parser_names_buffer = ['Rscript', r_file,input_name,'-c','0','--cut',cut,'--output',output_name]
parser_names_story = ['Rscript', r_file,input_name,'-c','1','--cut',cut,'--output',output_name]

################ Executes the lexicoder with the given parameters ################
subprocess.run(parser_names_buffer)
subprocess.run(parser_names_story)


##### Reading R code's output

PR_dfmat_buffer = pd.read_csv('../data_new/'+output_name+'_original_buffer.csv')
PR_dfmat_texts = pd.read_csv('../data_new/'+output_name+'_original_full_text.csv')

# Lemmatized lexicoder
lemm_PR_dfmat_texts = pd.read_csv('../data_new/'+output_name+'_lemm_full_text.csv')
lemm_PR_dfmat_buffer = pd.read_csv('../data_new/'+output_name+'_lemm_buffer.csv')

lemm_e_PR_dfmat_texts = pd.read_csv('../data_new/'+output_name+'_lemm_e_full_text.csv')
lemm_e_PR_dfmat_buffer = pd.read_csv('../data_new/'+output_name+'_lemm_e_buffer.csv')


def merge_dfs(df1,df2,ignore_cols = ['document','id','Agency']):
    """ merges multiple dataframes into one. 
    @ df1 & df2: dataframes with the exact same column names 
    @ ignore: list of columns to ignore 
    """

    df_out = df1.copy()

    for col in df_out:
        if col not in ignore_cols:
            df_out[col] = list(zip(df1[col],df2[col]))
    
    return df_out


original_df =  merge_dfs(PR_dfmat_texts, PR_dfmat_buffer)

#lemmatized
lemm_df = merge_dfs(lemm_PR_dfmat_texts,lemm_PR_dfmat_buffer)
lemm_e_df = merge_dfs(lemm_e_PR_dfmat_texts,lemm_e_PR_dfmat_buffer)

original_df.to_csv('../data_new/'+output_name+'_original_final.csv')
lemm_df.to_csv('../data_new/'+output_name+'_lemm_final.csv')
lemm_e_df.to_csv('../data_new/'+output_name+'_lemm_e_final.csv')



