
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import numpy as np

# BERT
from simpletransformers.language_representation import RepresentationModel

# LSTM
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling1D,Dense,Dropout,Activation



def data_preparation(df):
    
    raw_data = df.copy()

    raw_data_toxic = raw_data[raw_data['toxic'] == 1]


    raw_data['toxic_flag_temp'] = raw_data['toxic'] + raw_data['severe_toxic'] + raw_data['obscene'] + raw_data['threat'] + raw_data['insult'] + raw_data['identity_hate']

    raw_data['toxic_flag'] = raw_data['toxic_flag_temp'].map(lambda x: 0 if x == 0 else 1)


    raw_data.drop('toxic_flag_temp', axis = 1, inplace = True)


    raw_data['comment_text'] = raw_data['comment_text'].astype(str)


    def data_cleaning(raw_data):
        raw_data['comment_text_url_removed'] = raw_data['comment_text'].map(lambda x: re.sub('http\\S+', '', x))

        raw_data['comment_text_html_removed'] = raw_data['comment_text_url_removed'].map(lambda x: re.compile(r'<.*?>').sub('', x))

        raw_data['comment_text_cons_punc_removed'] = raw_data['comment_text_html_removed'].map(lambda x: re.compile(r'([.,/#!$%^&*;:{}=_`~()-])[.,/#!$%^&*;:{}=_`~()-]+').sub(r'\\1', x))

        raw_data['comment_text_cleaned'] = raw_data['comment_text_cons_punc_removed'].map(lambda x: re.sub('(\\r\\n?|\\n)+|\\s+', ' ', x))

        raw_data['comment_text_cleaned'] = raw_data['comment_text_cleaned'].str.replace('\\d+', '')

        raw_data['comment_text_cleaned'] = raw_data['comment_text_cleaned'].map(lambda x: x.lower())

        return raw_data


    raw_data_cleaned = data_cleaning(raw_data)


    raw_data_cleaned_new = raw_data_cleaned[['id', 'comment_text_cleaned', 'toxic_flag']]



    # # Undersampling



    s1 = raw_data_cleaned_new[raw_data_cleaned_new['toxic_flag'] == 0].shape[0]
    df_label = pd.DataFrame(np.random.randint(1,10,size=(s1, 1)), columns=['l1'])


    raw_data_cleaned_new_NT = raw_data_cleaned_new[raw_data_cleaned_new['toxic_flag'] == 0]
    raw_data_cleaned_new_T = raw_data_cleaned_new[raw_data_cleaned_new['toxic_flag'] == 1]

    raw_data_cleaned_new_NT['temp_label'] = df_label['l1']


    raw_data_cleaned_new_T['temp_label']  = 1


    raw_data_cleaned_new = pd.concat([raw_data_cleaned_new_NT, raw_data_cleaned_new_T], axis = 0)


    raw_data_cleaned_new = raw_data_cleaned_new[raw_data_cleaned_new['temp_label'] == 1]


    raw_data_cleaned_new.drop('temp_label', axis = 1, inplace = True)


    raw_data_cleaned_new = raw_data_cleaned_new[raw_data_cleaned_new['comment_text_cleaned'] != '']


    raw_data_cleaned_new.to_csv('intermediate_data/undersampled_intermediate_data_final.csv', index = False)
    
    return raw_data_cleaned_new
