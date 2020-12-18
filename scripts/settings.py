"""
    Settings for running scripts
"""

#Source newspaper (Reforma or Universal)
SOURCE_NEWSPAPER = 'El Universal (Mexico)'
#SOURCE_NEWSPAPER = 'Reforma (Mexico)'
# Each source = new dataframe? or everything together? Then SOURCE_NEWSPAPER should be a list

#Number of sentences around target sentence to consider for sentiment analysis
SENTENCE_BUFFER_WINDOW = 1

DATA_RAW = '../data_raw/'
DATA_NEW = '../data_new/'

OUT_1 = 'all_unstable_articles.pkl'
OUT_2 = 'all_unstable_nodup_title_index.pkl'
PROCESSED_ARTICLES = 'df_comprehensive_REGEX.pkl'

SENTIMENTS_FILE = 'all_article_sentiments.pkl'
DAILY_FILE = 'daily_with_dict.pkl'
AGGREGATED = 'df_aggregated.pkl'
ANOMALY_SVM = 'df_with_svm.pkl'