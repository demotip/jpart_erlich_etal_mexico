
""" 
Code to create the daily dataframe with all articles. The final dataframe includes
sentiment and corruption words
"""


import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet 
import numpy as np
import settings
from tqdm import tqdm
tqdm.pandas()


stemmer = SnowballStemmer("spanish")
spanish_stopwords = set(stopwords.words('spanish'))


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


def getMatchText(indices, sentences):
    # NOTE: can only handle one sentence windows currently
    '''
    takes in list of indices and list of sentences, returns aggregation of sentences 
    at the indices in the text block +-1 sentence as a string
    '''
    match_text = ''
    match_sentences_indices = []
    
    for index in indices:
        if len(sentences)==1: # if only sentence
            match_sentences_indices.append(index)            
        elif index == 0: # if first sentence
            match_sentences_indices.append(index)
            match_sentences_indices.append(index+1)
        elif index == len(sentences) - 1: # if last sentence
            match_sentences_indices.append(index)
            match_sentences_indices.append(index-1)
        else:
            match_sentences_indices.append(index)
            match_sentences_indices.append(index+1)
            match_sentences_indices.append(index-1)
    
    # remove duplicates        
    match_sentences_indices = list(set(match_sentences_indices))
    
    for index in match_sentences_indices:
        match_text += sentences[index]        
    return match_text
        

def appendMatchText(sep_df):
    '''
    get's matching text from story sentences indices, appends it to appropriate row
    returns df with appended text
    
    '''
    match_text  = []
    for i, row in sep_df.iterrows():
    
        indices = row.buffered_story_sentence_index
        sentences = getMatchText(indices, row.story_sentences)
        match_text.append(sentences)
    
    sep_df['story_sentences'] = match_text
    
    return sep_df

def get_first_pages(sep_df):
    '''
    gets the number of first page articles for a given agency/day pair    
    '''
    sep_df['section'] = sep_df['section'].fillna('not specified')
    sep_df['num_first_page'] = 0
    sep_df.loc[sep_df['section'].str.contains('(?:^|\W)SECTION: PRIMERA; Pág. 1(?:$|\W)',regex = True),'num_first_page'] = 1
    
    return sep_df

def addNonArticleRows(sep_df, avg_sentiment_df, datelist, entity_names, valid_keys,source_names,sent_df):
    '''
    adds to our agency_day dataframe the rows of agencies that did not have an 
    article written about them on that day
    
    the dataframe returned by the function has a row for each agency for each day
    that we have data on (2005-01-01 - 2016-12-31)
    '''
    entity_day_data = []

    for newspaper in source_names:
        for index, day in tqdm(enumerate(datelist)):
            for entity in entity_names:
                sentiment_key = (day, entity)
                if sentiment_key in valid_keys: # we know the agency has at least one article mention on this day
                    
                    #newspaper = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)].source.values
                    article_id = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].id.values
                    story_index = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].story_sentence_index.values
                    article_title = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].title.values
                    article_buffer = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].buffered_story_sentence_index.values
                    full_text = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].story.values
                    
                    num_first_page = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].num_first_page.values


                    match_text = sep_df.loc[(sep_df['agencies']==entity) & (sep_df['date']==day)& (sep_df['source']==newspaper)].story_sentences.values

                    match_text_count = 0
                    
                    num_articles = len(full_text)
                    
                    
                    # Getting the sentiments from the lexicoder
                    
                    if len(article_id) > 0:
                        new_sent_story = []
                        new_sent_buffer = []
                        for sent_id in article_id:
                            new_sent = sent_df.loc[(sent_id,entity),'log_sentiment']
                            new_sent = ast.literal_eval(new_sent)
                            new_sent_story.append(new_sent[0])
                            new_sent_buffer.append(new_sent[1])
                    else:
                        new_sent = 0
                        new_sent = 0
                        new_sent_story = 0
                        new_sent_buffer = 0
                    
                    
                    
                    
                    
                    
                    
                    if num_articles > 0:
                        full_text_sentiment = avg_sentiment_df.loc[day, entity]['full_text_senti']
                        match_text_sentiment = avg_sentiment_df.loc[day, entity]['matching_text_senti']
                    else: full_text_sentiment,match_text_sentiment = 0,0

                    for text in match_text:                    
                        match_text_count+=len(text.split())

                    entity_day = [day, entity,newspaper,num_first_page,article_id,article_title, full_text, match_text, match_text_count, num_articles,
                    story_index,article_buffer,match_text_sentiment,full_text_sentiment,new_sent_story,new_sent_buffer]
                else:
                    entity_day = [day, entity, newspaper, 0, None, None, None, None, None, None, None, None,None,None,None]

                entity_day_data.append(entity_day)

    entity_day_df = pd.DataFrame(data = entity_day_data)
    entity_day_df.columns = ['date', 'agency','newspaper','num_first_page','article_id', 'title', 'story', 'story_sentences',
    'num_sentences','num_articles','story_sentence_index', 'buffered_story_sentence_index', 'avg_senti_buffer','avg_senti_article','lexi_senti_story','lexi_senti_buffer']
    return entity_day_df


def buildAgencyDayDF(article_df,sent_df):
    
    '''
    Creates the daily dataframe. This dataframe contains a row for each agency-newspaper-day triple.
    '''    
    
    source_names = ['Reforma (Mexico)', 'El Universal (Mexico)']
    
    sep_df = sepMultiEntityRows(article_df)
    sep_df = fix_sentences(sep_df)
    sep_df = get_first_pages(sep_df)
    
    
    entity_names = list(set(sep_df['agencies'].values.tolist()))
    
    print('df separated\n')
    sep_df_w_text = appendMatchText(sep_df)
    print('text appended\n')

    #sep_df_w_text = sep_df_w_text.drop(columns = ['buffered_story_sentence_index', 'title'])
    
    

    ##### can delete this renaming line after naming conventions are set ######
    sep_df_w_text = sep_df_w_text.rename(columns={'agencies': 'entity', 'senti_avg_per_agency': 'matching_text_senti', 
                                                  'senti_full_article': 'full_text_senti', 'story' : 'full_text', 
                                                  'story_sentences':'matching_text' })
    ##########################################################################
    datelist = pd.date_range('2005-01-01', '2016-12-31').tolist()  # contains every day in dataframe
    datelist = [date.to_pydatetime().date() for date in datelist]
    
    #avg_sentiment_df = sep_df_w_text.groupby(['date','entity']).mean().groupby(['date','entity'])['matching_text_senti', 'full_text_senti'].mean()

    avg_sentiment_df = sep_df_w_text.groupby(['date','entity']).agg({'num_first_page':'sum','matching_text_senti' : lambda x: list(x),'full_text_senti' : lambda x: list(x),'id' : lambda x:list(x)})    
   
    # has every combo of date-agency where article was written about agency on that date
    valid_keys = [(key[0].date(), key[1]) for key in avg_sentiment_df.index.values]
    
    print('valid keys created \n')
    
    entity_day_df = addNonArticleRows(sep_df, avg_sentiment_df, datelist, entity_names, valid_keys,source_names,sent_df)
    
    print('added non article rows\n')

    entity_day_df.set_index(['date','agency','newspaper'],inplace = True)
    entity_day_df.sort_values(['date','agency','newspaper'],inplace = True)
    
    return entity_day_df




def getSynonyms(word_list, depth, stemmer, stem=False):
    
    '''
    gets nltk synset synonyms for word list
    args:
        word_list: list of words 
        depth: how deep the synonyms go, e.g. if depth is 2 we get synonyms and 
               the synonyms of the synonyms
        stemmer: stemming function (used with stemmer.stem('myWord'))
        stem: Boolean. If true, stem each word in the synonym list
    '''
    
    syn_lists = [word_list]
    for i in range(depth):
        
        new_set = []
        for word in syn_lists[-1]:

            for synset in wordnet.synsets(word, lang='spa'):
 
                for synword in synset.lemmas(lang='spa'):
                    new_set.append(synword.name())        
        
        syn_lists.append(new_set)
              
    final_set = []
    for synonyms in syn_lists:
        for word in synonyms:
            final_set.append(word)
             
    
    final_set = list(set(final_set))
            
    final_set = [word.replace('_', ' ') for word in final_set]      
    
    if stem:
        final_set = [stemmer.stem(word) for word in final_set]
    
    return final_set
                    

def buildCorruptionDict(text_list, wordlist):
    '''
    takes either anomaly_df.all_text or anomaly_df.matching_text and a word list
    for each anomaly, build the corruption-word dict, return list of dicts

    
    '''
    
    
    dict_list = []
    for i, text in enumerate(text_list):
        term_dict = {}
        if text is None:
            dict_list.append(None)
        else:
        
            text_str = ' '.join(text)
            for word in wordlist:
                
                term_dict[word]=text_str.count(word)
    
            
            dict_list.append(term_dict)
        
        
    return dict_list


def dict_to_columns(df,dict_names):
    '''
    transform each word in the corruption dict into a column
    
    '''
    update_names = []

    for k,dict_name in enumerate(dict_names):
        all_dict = pd.DataFrame({dict_name:df[dict_name]})
        has_dict = all_dict[~all_dict[dict_name].isnull()]
        no_dict = all_dict[all_dict[dict_name].isnull()]
        has_dict = pd.DataFrame(has_dict[dict_name].values.tolist(), index=has_dict.index).fillna(0)
        final = pd.concat([has_dict,no_dict],sort=True).sort_index()
        final = final.drop(columns = dict_name)
        names = list(final)
        new_names = ['v'+str(k)+'_'+column for column in names]
        final.columns = new_names
        df = pd.merge(df,final,left_index = True,right_index = True)
        update_names.append('v'+str(k)+'_'+dict_name)

    
    upd_dict = dict(zip(dict_names, update_names))
    
    df.rename(columns=upd_dict, inplace=True)

    return df


############################

corruption_words = ['escándalo','impunidad','soborno','sobornar','irregularidad', 'indagatorio', 'incumplimiento', 
                         'conflicto de intereses', 'indebido','indebidamente','malversar','malversación', 'corrupción', 
                         'corrupto', 'fraude', 'complicidad','negligencia', 'anomalía','desfalco','despilfarro', 
                         'desviar','desviación','enriquecimient']




corruption_words2 = ['corrupción','corrupt','escándal','escandal','impunidad','soborn','irregularidad', 'indagatoria',
                        'incumpl','conflicto de intereses','fraud','indebid','malvers', 'complicidad','negligen', 'anomalía',
                        'desfalc','despilfarr','desví','enriquecimient']



#######################



article_df = pd.read_pickle(settings.DATA_NEW+settings.SENTIMENTS_FILE)

sent_df = pd.read_csv(settings.DATA_NEW+'final_original_final.csv')
sent_df = sent_df.set_index(['id','Agency'])

df_daily = buildAgencyDayDF(article_df,sent_df)


words = getSynonyms(corruption_words, 1, stemmer, False)

######### can build as many dicts as you'd like
corruption_dict = buildCorruptionDict(df_daily.story, words)
corruption_dict_buffer = buildCorruptionDict(df_daily.story_sentences, words)

corruption_dict2 = buildCorruptionDict(df_daily.story, corruption_words2)
corruption_dict_buffer2 = buildCorruptionDict(df_daily.story_sentences, corruption_words2)

df_daily['corrupt_words_articles_syn'] = corruption_dict
df_daily['corrupt_words_buffer_syn'] = corruption_dict_buffer

df_daily['corrupt_words_articles_dan'] = corruption_dict2
df_daily['corrupt_words_buffer_dan'] = corruption_dict_buffer2



dict_names = ['corrupt_words_articles_syn','corrupt_words_buffer_syn','corrupt_words_articles_dan','corrupt_words_buffer_dan']

df_final = dict_to_columns(df_daily,dict_names)


df_final.reset_index(inplace = True) #reseting index in order to load it on R 
df_final['num_first_page'] = df_final['num_first_page'].apply(np.sum) #fixing the values of num_first_page




df_final.to_pickle(settings.DATA_NEW+'daily_with_lexi_sent.pkl')
