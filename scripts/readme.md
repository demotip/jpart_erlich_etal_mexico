# Scripts
Scripts for processing the Newspaper Articles dataset


## Description of files

### `04-anomaly_detection.py` : Performs anomaly detection using Generalized ESD

- In: '../data_new/df_comprehensive_REGEX.pkl' / ***Dataframe with stories and other info (including column with agencies that were cited)***
- Out: "../data_new/anomaly_date_df.pkl" / ***Dataframe with agencies and anomaly periods***


### `04-sentiment_ML.py` : Performs sentiment analysis on the article and buffer text, based on https://github.com/aylliote/senti-py

- In: '../data_new/df_comprehensive_REGEX.pkl' / ***Dataframe with stories and other info (including column with agencies that were cited)***
- Out: "../data_new/all_article_sentiments.pkl" / ***Dataframe with the sentiment of each story and buffer***


### `04-burstiness.py` : Performs burtiness detection using Kleinberg's algorithm

- In: '../data_new/df_comprehensive_REGEX.pkl' / ***Dataframe with stories and other info (including column with agencies that were cited)***
- Out: "../data_new/burstiness_date_df.pkl" / ***Dataframe with agencies and bursty periods***


### `04-sentiment_lexicoder.py` : Performs sentiment analysis on the article and buffer text, based on spanish lexicoder

- In: '../data_new/all_article_sentiments.pkl' / ***Dataframe with the sentiment of each story and buffer***
- Out: "../data_new/'+output_name+'_original_final.csv" , lemmatized and lemmatized extended versions / ***Dataframe with the sentiment using the normal lexicoder, the lemmatized and the lemmatized extended***


### `05-create_daily.py` : Creates the daily dataframe with all articles. The final dataframe includes sentiment and corruption words

- In: '../data_new/all_article_sentiments.pkl' / ***Dataframe with the sentiment of each story and buffer***
- Out:  '../data_new/daily_with_dict.pkl' / ***Dataframe with one row for each agency-newspaper-day triple***


### `06-SVMWords.py` : Uses a SVM to get the 10 most important words for distinguishing each anomaly

- In: '../data_new/daily_with_dict.pkl' and '../data_new/anomaly_date_df.pkl' / ***Daily and anomaly df (see above)***
- Out:  '../data_new/anomaly_svm.pkl' ***Dataframe with the anomaly information and the SVM words that most distinguish an anomaly story from a normal one***



### `07-merge_info.py` : Aggregates the daily information based on the anomalies dates


- In: '../data_new/daily_with_dict.pkl' and '../data_new/anomaly_svm.pkl' / ***Daily and SVM df (see above)***
- Out:  '../data_new/aggregated_df.pkl' / ***Aggregated DF with all information based on the anomaly dates***


### `08-anomalies_to_request.py` : Maps requests based on the SVM words obtained from the anomalies


- In: "../data_new/anomaly_date_df.pkl", '../data_new/anomaly_svm.pkl' and '../data_raw/mongo_requests.csv' / ***Dataframe with the anomalies, with svm words and the requests***
- Out:  '../data_new/svm_requests.csv' / ***Dataframe with the requests that matched at least one of the SVM words during a given period (usually 2 weeks before and 2 weeks after the anomaly period)***


## Other files

### `Agency_newspaper_metrics.r` : R code that performs sentiment analysis based on the lexicoder

- In: "../data_new/all_agency_news_with_buff_mod.csv" and '../data_raw/lemmatized*' / ***Dataframe with one column for the stories and another for the buffer and the lexicoder dictionary***
- Out:  sentiment analysis dataframes for buffer and story separately that are later merged by `04-sentiment_lexicoder.py` 

**Inside the auxiliary codes folder**

### `export_articles.py` : Get the top articles based on the SVM words


- In: A dataframe with SVM words, an anomaly dataframe and the dataframe with all sentiments
- Out:  One txt file for each one of the anomalies with the articles ordered by svm words count

 ------------------------------------------------------------------------------------------------


![img](https://i.imgur.com/aJvsyxk.png)


The code to preprocess the data is not publicly available. For more information about the preprocessing and the data please contact the authors.


## Description of some additional parameters

- Anomaly detection (`04A-anomaly_detection.py`)
The anomaly detection is based on the the generalized extreme Studentized deviate (GESD), which is used to detect one or more outliers in a univariate data set that follows an approximately normal distribution.
The function has two parameters: 'MAXOUT' - the maximum number of anomalies (upper bound for the detection) and 'alpha' - the significance level for the t-distribution in the test

- Burstiness detection (`04C-burstiness.py`)
Performs burtiness detection using Kleinberg's algorithm (https://www.cs.cornell.edu/home/kleinber/bhs.pdf). It identifies time periods in which a target event is uncharacteristically frequent, or “bursty.” 
The function has three parameters:
's' - The goodness of fit between the observed proportion and the expected probability of each state. The closer the observed proportion is to the expected probability of a state, the more likely the system is in that state (bursty or baseline). Can be interpreted as the "distance between states". The higher this parameter, ther shorter are the burts.
'gamma' - The difficulty of transitioning from the baseline state to the bursty state. There’s a cost associated with entering a higher state, but no cost associated with staying in the same state or returning to  a lower state. The transition cost, denoted by tau, therefore equals zero when transitioning to a lower state or staying in the same state. If gamma increases, we get fewer and shorter bursts.
'smooth_win' - Smooth window to smooth the data, use a odd number

