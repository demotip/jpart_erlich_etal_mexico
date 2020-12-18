"""
Maps the requests based on the svm words obtained from the anomalies
"""
import spacy
import itertools
import pandas as pd
import numpy as np
import pickle
import os
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
stemmer = SnowballStemmer("spanish")
spanish_stopwords = set(stopwords.words('spanish')) 
from datetime import timedelta
import ast


def select_requests(df_idx,df_info,agency_dict,weeks_before,weeks_later):
    """
    select the requests based on the anomaly date and agency
    Arguments:
        - df_idx: anomaly dataframe
        - df_info: requests dataframe
        - agency_dict: dictionary with the corresponding agency names
        - weeks_before: number of weeks before the anomaly started that is considered when mapping the requests
        - weeks_later: number of weeks after the anomaly ended that is considered when mapping the requests


    """
    df = pd.DataFrame()
    for i,row in df_idx.iterrows():
        if type(agency_dict[row.Agency]) == str:
            _temp = df_info[df_info.DEPENDENCIA==agency_dict[row.Agency]][['FECHASOLICITUD','DESCRIPCIONSOLICITUD','OTROSDATOS','FOLIO','DEPENDENCIA']]
            _temp.FECHASOLICITUD = pd.to_datetime(_temp.FECHASOLICITUD)
            mask = (_temp.FECHASOLICITUD >= row['Start Date'] - pd.Timedelta(weeks_before, unit='w')) & (_temp.FECHASOLICITUD <= row['End Date'] + pd.Timedelta(weeks_later, unit='w'))
            _temp = _temp[mask]
            _temp.fillna(' ',inplace = True)
            _temp = _temp[_temp.DESCRIPCIONSOLICITUD != 'DESCRIPCIÓN SOLICITUD']
            _temp['Text'] = _temp.DESCRIPCIONSOLICITUD +' ' + _temp.OTROSDATOS
            _temp['Agency'] = row.Agency
            _temp['Date'] = _temp.FECHASOLICITUD
            _temp['Anomaly_id'] = i
            _temp['Start_Date'] = row['Start Date']
            _temp['End_Date'] = row['End Date']
            _temp['Folio'] = _temp.FOLIO
            _temp['Dependencia'] = _temp.DEPENDENCIA
            df = pd.concat([df,_temp[['Folio','Dependencia','Agency','Text','Date','Anomaly_id','Start_Date','End_Date']]])
        
        else:
            for name in agency_dict[row.Agency]:
                _temp = df_info[df_info.DEPENDENCIA==name][['FECHASOLICITUD','DESCRIPCIONSOLICITUD','OTROSDATOS','FOLIO','DEPENDENCIA']]
                _temp.FECHASOLICITUD = pd.to_datetime(_temp.FECHASOLICITUD)
                mask = (_temp.FECHASOLICITUD >= row['Start Date'] - pd.Timedelta(weeks_before, unit='w')) & (_temp.FECHASOLICITUD <= row['End Date'] + pd.Timedelta(weeks_later, unit='w'))
                _temp = _temp[mask]
                _temp.fillna(' ',inplace = True)
                _temp = _temp[_temp.DESCRIPCIONSOLICITUD != 'DESCRIPCIÓN SOLICITUD']
                _temp['Text'] = _temp.DESCRIPCIONSOLICITUD +' ' + _temp.OTROSDATOS
                _temp['Agency'] = row.Agency
                _temp['Date'] = _temp.FECHASOLICITUD
                _temp['Anomaly_id'] = i
                _temp['Start_Date'] = row['Start Date']
                _temp['End_Date'] = row['End Date']
                _temp['Folio'] = _temp.FOLIO
                _temp['Dependencia'] = _temp.DEPENDENCIA
                df = pd.concat([df,_temp[['Folio','Dependencia','Agency','Text','Date','Anomaly_id','Start_Date','End_Date']]])      

        
    return df


def buildSVMDict(text_list, wordlist):
    '''
    takes either df.all_text or df.buffer and a word list
    for each anomaly, build a dict based on the occurancies on the wordlistr, return list of dicts

    
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


def countSVM(df,svm_words):
    """
    Takes the processed request dataframe and the svm words and return a dataframe with the requests that matched at least one
    svm word. 
    """
    
    df_out = pd.DataFrame()
    
    for anomaly_id,words in enumerate(svm_words):
        words_list = ast.literal_eval(words)
        words_list = [n.strip() for n in words_list]
        svm_stem = [stemmer.stem(word) for word in words_list]
        subset_df = df[df.Anomaly_id == anomaly_id]
        svm_dict = buildSVMDict(subset_df.Text,svm_stem)
        words_count = [sum(d.values()) for d in svm_dict]
        words_count = np.array(words_count)
        df_select = subset_df[words_count>0]
        df_select['svm_count'] = words_count[words_count>0]
        df_out = pd.concat([df_out,df_select])
        
    return df_out




df_anom = pd.read_pickle('../data_new/anomaly_date_df_100.1.pkl')
df_info = pd.read_csv('../data_raw/mongo_requests.csv')
df_svm = pd.read_csv('../data_new/aggregated_anomaly_svm_stop_max10_alpha10.csv')


agency_dict = {'INM' : 'INSTITUTO NACIONAL DE MIGRACIÓN','IMSS':'INSTITUTO MEXICANO DEL SEGURO SOCIAL','SEGOB':['SECRETARÍA DE GOBERNACIÓN','SECRETARÍA DE GOBERNACIÓN (INCLUYE LA ENTONCES SECRETARÍA DE SEGURIDAD PÚBLICA)'],'SRE':'SECRETARÍA DE RELACIONES EXTERIORES',
            'CONAGUA':'COMISIÓN NACIONAL DEL AGUA','SENER':'SECRETARÍA DE ENERGÍA',
            'COFEPRIS':'COMISIÓN FEDERAL PARA LA PROTECCIÓN CONTRA RIESGOS SANITARIOS','SAGARPA':'SECRETARÍA DE DESARROLLO SOCIAL',
            'PEMEX':['PEMEX EXPLORACIÓN Y PRODUCCIÓN','PEMEX REFINACIÓN','PEMEX GAS Y PETROQUÍMICA BÁSICA','PEMEX PETROQUÍMICA'],'SHCP':'SECRETARÍA DE HACIENDA Y CRÉDITO PÚBLICO',
            'CFE':'COMISIÓN FEDERAL DE ELECTRICIDAD','SEDENA':'SECRETARÍA DE LA DEFENSA NACIONAL','IMPI':'INSTITUTO MEXICANO DE LA PROPIEDAD INDUSTRIAL',
            'SEDESOL':'SECRETARÍA DE DESARROLLO SOCIAL','SFP':'SECRETARÍA DE LA FUNCIÓN PÚBLICA','SEP':'SECRETARÍA DE EDUCACIÓN PÚBLICA','SSA':'SECRETARÍA DE SALUD',
            'SEMARNAT':'SECRETARÍA DE MEDIO AMBIENTE Y RECURSOS NATURALES','SSP':'SECRETARÍA DE SEGURIDAD PÚBLICA','SEECO':'SECRETARÍA DE ECONOMÍA',
            'SCT':'SECRETARÍA DE COMUNICACIONES Y TRANSPORTES','PGR':'PROCURADURÍA GENERAL DE LA REPÚBLICA'}

df = select_requests(df_anom,df_info,agency_dict,2,2)

df = df.reset_index()
df = df.drop(columns = 'index')
svm_words = df_svm.SVM_words

df_final = countSVM(df,svm_words)

df_final.to_csv('../data_new/svm_requests.csv')