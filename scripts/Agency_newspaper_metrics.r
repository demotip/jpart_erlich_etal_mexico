### *** Agency newspaper metrics *** ### 

### Setup ### 

#libraries 
library(readr)
library(utf8)
library(quanteda)
library(argparser,quietly=TRUE)
library(spacyr)
spacy_initialize("es_core_news_md")


######################### Parser arguments ##############################

p <- arg_parser("Parameters for the sentiment analysis")

p <- add_argument(p,"input",help="input file")
p <- add_argument(p,"--column_num",help="column to perform SA, 0 buffer, 1 full story",type = "numeric", default = 0)
p <- add_argument(p,"--cut",help="number of cuts",type = "numeric", default = 0)
p <- add_argument(p,"--output",help="output file",default = 'metrics_agency_news')

argv <- parse_args(p)


#*****************************************************************

# Read documents
#PATH_IN = "../data_raw/agency_news_with_buff_mod.csv"

PATH_IN <- argv$input
df <- read_csv(PATH_IN)

# ********************************************************************************************************

### Corpora ### 

# Compile the corpus of raw texts (full texts and buffer)

if(argv$column_num == 0){
  column = "Buffer_Text"
  lemmatized_column = "lemmatized_buffer"
}else{
  column = 'Story'
  lemmatized_column = "lemmatized_full_text"
}


raw_texts <- c(df[[column]])  
raw_corp <- corpus(raw_texts)

# COmpile the corpus of lemmatized texts 
lemma_texts <- c(df[[lemmatized_column]]) # lemmatized version of the column to work with
lemma_corp <- corpus(lemma_texts)

# # check corpora
# print("raw corpus")
# summary(raw_corp, 10) 
# print("lemmatized corpus")
# summary(lemma_corp, 10)

# **********************************************************************************************************

### Dictionaries ### 

# PATHS

Proksch_es_PATH = "../data_raw/extendeddict_es.lc3" # Proksch

# The dictionary options below are the different versions of the lexicoder according to the cut parameter. 
# Choosing a different çut' version of the dictionary dictionary will change the outputs. 
# See Lexicoder_lemmatizer_es_v1.py


if (argv$cut == 0){
 Lemm_Proksch_es_PATH = "../data_new/lemmatized_extendeddict_es_0.lc3" # Proksch autotranslated , cut=0
 Lemm_e_Proksch_es_PATH = "../data_new/lemmatized_extendeddict_es_e_0.lc3" # Proksch extended autotranslated, cut=0
  
}else if (argv$cut == 1){
 Lemm_Proksch_es_PATH = "../data_new/lemmatized_extendeddict_es_1.lc3" # Proksch autotranslated , cut=1
 Lemm_e_Proksch_es_PATH = "../data_new/lemmatized_extendeddict_es_e_1.lc3" # Proksch extended autotranslated, cut=1

}else{
 Lemm_Proksch_es_PATH = "../data_new/lemmatized_extendeddict_es_2.lc3" # Proksch autotranslated , cut=2
 Lemm_e_Proksch_es_PATH = "../data_new/lemmatized_extendeddict_es_e_2.lc3" # Proksch extended autotranslated, cut=2
  
}





# Load the ditionaries
PR_dict <- dictionary(file = Proksch_es_PATH, format="lexicoder", enc="unicode")
lemm_PR_dict <- dictionary(file = Lemm_Proksch_es_PATH, format="lexicoder",enc="unicode")
lemm_e_PR_dict <- dictionary(file = Lemm_e_Proksch_es_PATH, format="lexicoder",enc="unicode")

# **********************************************************************************************************

### dfm objects : Tokenization and token lookup ### 

## 1. Token creations 

# 1.1 Raw tokens

raw_toks <- as.tokens(spacy_tokenize(raw_texts,
                                     remove_punct = TRUE, 
                                     remove_numbers = TRUE , 
                                     remove_symbols = TRUE , 
                                     tolower = TRUE, 
                                     ngrams=1:3))

# 1.2 Lemmatized tokens 

lemma_toks <- as.tokens(spacy_tokenize(lemma_texts, 
                                       remove_punct=TRUE,
                                       remove_numbers = TRUE, 
                                       remove_symbols = TRUE, 
                                       tolower = TRUE,
                                       ngrams=1:3))

## 2. Tokens lookup 

# 2.1 Raw tokens on raw text

PR_toks_l <- tokens_lookup(raw_toks , dictionary = PR_dict, valuetype = "fixed", verbose=TRUE,  nomatch = "NONE")
lemm_PR_toks_l <- tokens_lookup(lemma_toks , dictionary = lemm_PR_dict, valuetype = "fixed", verbose=TRUE,  nomatch = "NONE")
lemm_e_PR_toks_l <- tokens_lookup(lemma_toks , dictionary = lemm_e_PR_dict, valuetype = "fixed", verbose=TRUE,  nomatch = "NONE")

## 3. dfmats 

PR_dfmat <- dfm(PR_toks_l)
lemm_PR_dfmat <- dfm(lemm_PR_toks_l)
lemm_e_PR_dfmat <- dfm(lemm_e_PR_toks_l)

# 
# ## Output results  
# 
# sprintf("2. Proksch auto translated")
# print(head(PR_dfmat,20))
# sprintf("6. Lemmatized autotranslated Proksch")
# print(head(lemm_PR_dfmat, 20))
# sprintf("7. Lemmatized extended autotranslated Proksch")
# print(head(lemm_e_PR_dfmat, 20))


# **********************************************************************************************************


### Custom Metrics functions ### 


net_tone <- function(num_pos_w, num_neg_w, num_all_w, scale_factor=1) { 
  return(
    (( num_pos_w/num_all_w ) - ( num_neg_w / num_all_w ))*scale_factor
  )
}

log_sentiment <- function(pos, neg){ 
  return(
    log((pos+0.5)/(neg+0.5))
  )
}



empiric_var <- function(pos, neg) { 
  return(
    1/(pos+0.5) + 1/(neg+0.5)
  )
}

# Empiric prediction boundaries 
raw_net_prediction <- function(net_sentiment) { 
  if(net_sentiment < -0.5){ 
    return(-1)
  }
  else if(net_sentiment > 0.5){ 
    return(1)
  }
  else(
    return(0)
  )
}

raw_log_prediction <- function(log_sentiment, tolerance=0.30) {
  if(log_sentiment < -tolerance){ 
    return(-1)
  }
  else if(log_sentiment > tolerance){ 
    return(1)
  }
  else(
    return(0)
  )
}


# **********************************************************************************************************

### Metrics calcualtions ### 


scale_factor = 1000000

## 1. Convert to data frame 

PR_dfmat_df <- convert(PR_dfmat, to = "data.frame") # 2
lemm_PR_dfmat_df <- convert(lemm_PR_dfmat, to = "data.frame") # 2
lemm_e_PR_dfmat_df <- convert(lemm_e_PR_dfmat, to = "data.frame") # 4

## 2. Calculate net tone 

PR_dfmat_df['net_tone'] <- net_tone(PR_dfmat_df[['+positive']],
                                    PR_dfmat_df[['+negative']], 
                                    sum(PR_dfmat_df[['+positive']],
                                        PR_dfmat_df[['+negative']],
                                        PR_dfmat_df[['none']]), 
                                    scale_factor = scale_factor
)


lemm_PR_dfmat_df['net_tone'] <- net_tone(lemm_PR_dfmat_df[['+positive']],
                                         lemm_PR_dfmat_df[['+negative']], 
                                         sum(lemm_PR_dfmat_df[['+positive']],
                                             lemm_PR_dfmat_df[['+negative']],
                                             lemm_PR_dfmat_df[['none']]), 
                                         scale_factor = scale_factor
)

lemm_e_PR_dfmat_df['net_tone'] <- net_tone(lemm_e_PR_dfmat_df[['+positive']] + lemm_e_PR_dfmat_df[['+neg_negative']],
                                           lemm_e_PR_dfmat_df[['+negative']] + lemm_e_PR_dfmat_df[['+neg_positive']],
                                           sum(lemm_e_PR_dfmat_df[['+positive']],
                                               lemm_e_PR_dfmat_df[['+neg_positive']],
                                               lemm_e_PR_dfmat_df[['+negative']],
                                               lemm_e_PR_dfmat_df[['+neg_negative']], 
                                               lemm_e_PR_dfmat_df[['none']]), 
                                           scale_factor = scale_factor
)

## 3. Log sentiment 

PR_dfmat_df['log_sentiment'] <- log_sentiment(PR_dfmat_df[['+positive']], PR_dfmat_df[['+negative']] )

lemm_PR_dfmat_df['log_sentiment'] <- log_sentiment(lemm_PR_dfmat_df[['+positive']],  lemm_PR_dfmat_df[['+negative']]  )

lemm_e_PR_dfmat_df['log_sentiment'] <- log_sentiment(lemm_e_PR_dfmat_df[['+positive']] + lemm_e_PR_dfmat_df[['+neg_negative']],
                                                     lemm_e_PR_dfmat_df[['+negative']] + lemm_e_PR_dfmat_df[['+neg_positive']] )

## 4. Calculate empirica variation 

PR_dfmat_df['empiric_var'] <- empiric_var(PR_dfmat_df[['+positive']], PR_dfmat_df[['+negative']] )
lemm_PR_dfmat_df['empiric_var'] <- empiric_var(lemm_PR_dfmat_df[['+positive']],  lemm_PR_dfmat_df[['+negative']]  )

lemm_e_PR_dfmat_df['empiric_var'] <- empiric_var(lemm_e_PR_dfmat_df[['+positive']] + lemm_e_PR_dfmat_df[['+neg_negative']],
                                                 lemm_e_PR_dfmat_df[['+negative']] + lemm_e_PR_dfmat_df[['+neg_positive']] )

## 5. Calculate empiric std

PR_dfmat_df['empiric_std'] <- sapply(PR_dfmat_df[['empiric_var']] , sqrt)
lemm_PR_dfmat_df['empiric_std'] <- sapply(lemm_PR_dfmat_df[['empiric_var']] , sqrt)
lemm_e_PR_dfmat_df['empiric_std'] <- sapply(lemm_e_PR_dfmat_df[['empiric_var']] , sqrt)

## 6. Exponential of log sentiment 
PR_dfmat_df['exp_sentiment'] <- sapply(PR_dfmat_df[['log_sentiment']] , exp)
lemm_PR_dfmat_df['exp_sentiment'] <- sapply(lemm_PR_dfmat_df[['log_sentiment']] , exp)
lemm_e_PR_dfmat_df['exp_sentiment'] <- sapply(lemm_e_PR_dfmat_df[['log_sentiment']] , exp)

## 7. raw_net_prediction 
PR_dfmat_df['raw_net_prediction'] <- sapply(PR_dfmat_df[['net_tone']] , raw_net_prediction)
lemm_PR_dfmat_df['raw_net_prediction'] <- sapply(lemm_PR_dfmat_df[['net_tone']] , raw_net_prediction)
lemm_e_PR_dfmat_df['raw_net_prediction'] <- sapply(lemm_e_PR_dfmat_df[['net_tone']] , raw_net_prediction)

## 8. raw_log_prediction 
PR_dfmat_df['raw_log_prediction'] <- sapply(PR_dfmat_df[['log_sentiment']] , raw_log_prediction)
lemm_PR_dfmat_df['raw_log_prediction'] <- sapply(lemm_PR_dfmat_df[['log_sentiment']] , raw_log_prediction)
lemm_e_PR_dfmat_df['raw_log_prediction'] <- sapply(lemm_e_PR_dfmat_df[['log_sentiment']] , raw_log_prediction)

## 9. raw_prediction 
PR_dfmat_df['raw_prediction'] <- sapply(PR_dfmat_df[['log_sentiment']] , sign)
lemm_PR_dfmat_df['raw_prediction'] <- sapply(lemm_PR_dfmat_df[['log_sentiment']] , sign)
lemm_e_PR_dfmat_df['raw_prediction'] <- sapply(lemm_e_PR_dfmat_df[['log_sentiment']] , sign)

## 10. add id's column 
PR_dfmat_df['id'] <- df['Article_ID']
lemm_PR_dfmat_df['id'] <- df['Article_ID']
lemm_e_PR_dfmat_df['id'] <- df['Article_ID']

## 10. add id's column 
PR_dfmat_df['Agency'] <- df['Agency']
lemm_PR_dfmat_df['Agency'] <- df['Agency']
lemm_e_PR_dfmat_df['Agency'] <- df['Agency']


# # Check metrics dfmats 
# sprintf("2. Proksch auto translated")
# print(head(PR_dfmat_df,10))
# sprintf("6. Lemmatized autotranslated Proksch")
# print(head(lemm_PR_dfmat_df, 10))
# sprintf("7. Lemmatized extended autotranslated Proksch")
# print(head(lemm_e_PR_dfmat_df, 10))


# *******************************************************************************************************************

### Export the results ### 

# NOTE: Depending on the version of the lexicoder used, please changee OUT PATHs accordingly

# TODO: Change output names
base_name <- argv$output
if(argv$column_num == 0){
  write.csv(PR_dfmat_df, file=paste("../data_new/",base_name,"_original_buffer.csv",sep = ''))
  write.csv(lemm_PR_dfmat_df, file=paste("../data_new/",base_name,"_lemm_buffer.csv",sep = ''))
  write.csv(lemm_e_PR_dfmat_df, file=paste("../data_new/",base_name,"_lemm_e_buffer.csv",sep = ''))
  
}else{
  write.csv(PR_dfmat_df, file=paste("../data_new/",base_name,"_original_full_text.csv",sep = ''))
  write.csv(lemm_PR_dfmat_df, file=paste("../data_new/",base_name,"_lemm_full_text.csv",sep = ''))
  write.csv(lemm_e_PR_dfmat_df, file=paste("../data_new/",base_name,"_lemm_e_full_text.csv",sep = ''))
  
  
}


