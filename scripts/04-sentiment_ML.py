"""
    Sentiment analysis.
    
    Class SentimentAnalyzer needs to be instantiated with the following:
        - df : a dataframe in the format of input_standard_dataframe
        - buffer: how many sentences around indentified sentence to take into account for sentiment analysis.
        - clf : Sentiment Analyzer.
            Requires this analyzer to have a function predict() which returns a double between 0 and 1.
        - REGEXES: Dictionary of { Agency_name : REGEX_EXPRESSION }. 
            This can be the same REGEX expression used to generate story_sentence_index.
            This is used to remove any mention of agency during sentiment analysis.
            
    Outputs:
        - article_sentiment_standard_dataframe (see readme) 
    
"""

import pandas as pd
from classifier import * #this is from github, spanish_sentiment_analysis
from tqdm import tqdm
import settings
import utils
import re
tqdm.pandas()


class SentimentAnalyzer():
    def __init__(
        self, 
        df = pd.read_pickle(settings.DATA_NEW+settings.PROCESSED_ARTICLES), 
        buffer = settings.SENTENCE_BUFFER_WINDOW,
        clf = SentimentClassifier(),
        REGEXES = utils.REGEXES
    ):


        self.df = df
        self.clf = clf
        self.buffer = buffer
        self.REGEXES = REGEXES
        
    def __series_wrapper_apply_buffer(self, series):
        """
        Private.
        This function takes the story_sentence_index dictionary and removes duplicates when buffer is applied.
        For example, if agency {PEMEX : [0,3,6,8,10]} was identified, and buffer = 1.
        This function returns {PEMEX : [0,1,2,3,4,5,6,7,8,9,10]}
        
        Returns a dictionary:
                {
                    agency_1 : [sentence_idx_1, sentence_idx_2, ...]
                    agency_2 : [sentence_idx_1, ...]
                    ...
                }
        """
        
        d = {}
        for agency, sentence_idxs in series.story_sentence_index.items():
            tmp = set()
            for sentence_idx in sentence_idxs:
                #Compute start and end indices
                start = max(0, sentence_idx - self.buffer)
                end = min(len(series.story_sentences) - 1, sentence_idx + self.buffer)
                #Insert everything into the set
                for i in range(start, end+1):
                    tmp.add(i)
                    
            d[agency] = sorted(list(tmp))
        return d
            
    def apply_buffer(self):
        """
        Creates a new column called buffered_story_sentence_index
        """
        print("Processing Sentence buffers...")
        self.df['buffered_story_sentence_index'] = self.df.progress_apply(self.__series_wrapper_apply_buffer, axis=1)
        
        
    def get_sentence_sentiment(self, sentence_idx, story_sentences, agency_name):
        """
        Returns the score of one sentence
        
        Arguments:
            -sentence_idx: index of sentence
            -story_sentences: list of sentences
            -agency_name: name of agency
        """
        

        #Buffer already applied.
        sample = story_sentences[sentence_idx].lower()
        
        # We need to remove the name of the agency because of prior sentiment attached.
        # There should only be one regex. We loop through all just in case.
        rg = re.compile(self.REGEXES[agency_name])
        sample = re.sub(rg, '', sample)
        return self.clf.predict(sample)
    
    def __series_wrapper_fulltext(self, series):
        #concatenate all sentences together
        sample = ' '.join(series.story_sentences).lower()
        
        #remove all mentions of agencies
        for agency_name in series.agencies:
            rg = re.compile(self.REGEXES[agency_name])
            sample = re.sub(rg, '', sample)
            
        #predict and return
        return self.clf.predict(sample)

    def __series_wrapper_sentence(self, series):
        """
            Should not get called directly. Private.
            Wraps get_sentence_sentiment so it can run on series.
            Runs on buffered_story_sentence_index.
            Returns a dictionary:
                {
                    agency_1 : [score1, score2, ...]
                    agency_2 : [score1, ...]
                    ...
                }
        """
        d = {}
        for agency, sentence_idxs in series.buffered_story_sentence_index.items():
            d[agency] = list(map(lambda x : self.get_sentence_sentiment(
                x, series.story_sentences, agency), sentence_idxs))
            
        return d

    def compute_sentiments(self, testing=False):
        """
        Appends 3 series to self.df:
            - senti_per_agency : Sentiment of sentence and buffer per agency
            - senti_avg_per_agency : Average sentiment per agency for article
            - senti_full_article: Sentiment for entire article
        """
        
        #apply Buffer
        self.apply_buffer()
        
        if testing:
            self.df = self.df.iloc[:20,:]
        
        #Getting every sentiment for every sentence
        print("\nProcessing sentence sentiment scores ...\n")
        senti_per_agency = self.df.progress_apply(self.__series_wrapper_sentence, axis=1)
        
        #helper function. Calculates average for each dictionary
        def _helper(d):
            new_dict = {}
            for agency, scores in d.items():
                new_dict[agency] = sum(d[agency])/len(d[agency])
            return new_dict
                
        #Calculates averages
        print("\nProcessing AVG sentence sentiment scores ...\n")
        senti_avg_per_agency = senti_per_agency.apply(lambda d : _helper(d))
        
        #Calculates entire text
        print("\nProcessing full document sentiment scores ...\n")
        senti_full_article = self.df.progress_apply(self.__series_wrapper_fulltext, axis=1)
        
        #Append new columns
        self.df['senti_per_agency'] = senti_per_agency
        self.df['senti_avg_per_agency'] = senti_avg_per_agency
        self.df['senti_full_article'] = senti_full_article
        
        #Return new columns
        return pd.DataFrame(data={
            'id' : self.df['id'],
            'senti_per_agency' : senti_per_agency,
            'senti_avg_per_agency' : senti_avg_per_agency,
            'senti_full_article' : senti_full_article
        })
              

def main():
    sa = SentimentAnalyzer(buffer=1)
    sa.compute_sentiments()
    sa.df.to_pickle(settings.DATA_NEW+settings.SENTIMENTS_FILE)


if __name__ == '__main__':
	main()
