
import pandas as pd
import numpy as np
import time, requests, json
import os
import sys
import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import gensim
import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer
from simpletransformers.language_representation import RepresentationModel

from empath import Empath




# ## Universal Sentence Encoder


# features_li_use = list(range(0,512))

pre = 'feat_'
features_li_use = []
for i in range(1, 513):
    features_li_use.append(pre + str(i))

def universal_sentence_encoder_func(data, feats_li = features_li_use):
    raw_data = data.copy()

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    raw_data[feats_li] = raw_data.apply(lambda x: embed(list(x['comment_text_cleaned']))[0], axis = 1, result_type = 'expand')
    
    raw_data.to_csv('intermediate_data/universal_sentence_encoder_features.csv', index = False)
    
    return raw_data, feats_li





# ## Sentence Transformer


features_li_st = list(range(0,384))

pre = 'feat_'
features_li_st = []
for i in range(1, 385):
    features_li_st.append(pre + str(i))

def sentence_transformer_func(data, feats_li = features_li_st):
    raw_data = data.copy()

    model_st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    raw_data[feats_li] = raw_data.apply(lambda x: model_st.encode(list(x['comment_text_cleaned']))[0], axis = 1, result_type = 'expand')
    
    raw_data.to_csv('intermediate_data/sentence_transformer_features.csv', index = False)
    
    return raw_data, feats_li





# ## TF-IDF


def tf_idf_func(raw_data):
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    df = raw_data.copy()


    def rm_stopwords_lam(t):
        text = [lemmatizer.lemmatize(word) for word in t.split() if not word in set(stopwords)]
        return text

    df['comment_text_cleaned_new'] = df['comment_text_cleaned'].map(lambda x: rm_stopwords_lam(x))

    def rm_short_wrds(t):
        new_li = ''
        for i in t:
            if len(i) >= 3:
                new_li = new_li + i + ' '
        return new_li

    df['comment_text_cleaned_new'] = df['comment_text_cleaned_new'].map(lambda x: rm_short_wrds(x))



    vectorizer = CountVectorizer(max_features = 500)

    X = vectorizer.fit_transform(df['comment_text_cleaned_new'])

    df_cnt_vec = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())

    vocab_li = list(df_cnt_vec.columns)

    df = df.reset_index(drop = True)

    tf_idf = TfidfVectorizer(vocabulary = vocab_li)

    X_train_tf = tf_idf.fit_transform(df['comment_text_cleaned_new'])

    df_tf_idf = pd.DataFrame(X_train_tf.toarray(), columns = tf_idf.get_feature_names())



    # right_index
    tf_idf_featured_data = pd.merge(df_tf_idf, df[['id', 'toxic_flag']], how = 'left', right_index = True, left_index = True)

    tf_idf_featured_data.to_csv('intermediate_data/tf_idf_featured_data.csv', index = False)
    
    return tf_idf_featured_data, vocab_li




# ## HurtLex


def hurt_lex_func(data):
    raw_data = data.copy()
    HL_VERSION = "1.2"
    categories = ["ps", "pa", "ddf", "ddp", "asf", "pr", "om", "qas"]
    LEN = len(categories)
    
    def read_lexicon(level, language):
        lexicon = dict()
        lexicon_filename = "hurtlex_EN.tsv"

        with open(lexicon_filename, encoding='utf-8') as f:

            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["level"]!=level:
                    continue
                if not row["lemma"] in lexicon:
                    lexicon[row["lemma"]] = np.zeros(2*LEN)
                if row["category"] in categories:
                    if level == "inclusive":
                        lexicon[row["lemma"]][LEN + categories.index(row["category"])] += 1
                    else:
                        lexicon[row["lemma"]][categories.index(row["category"])] += 1
        return lexicon

    def check_presence(lexicon, text):
        text = re.sub(r'[^a-zA-Z\s]+$', '', text)
        final_features = np.zeros(2*LEN)
        for k,v in lexicon.items():
            string = r"\b" + k.strip() + r"\b"
            all_matches = re.findall(string.strip(), text)
            for match in all_matches:
                lexicon_val = np.zeros(2*LEN)
                for item in lexicon:
                    if match in item:
                        lexicon_val = lexicon[item]
                final_features = np.add(final_features, lexicon_val)
        return final_features

    def process(text):
        return np.add(check_presence(conservative_lexicon, text), check_presence(inclusive_lexicon, text))

    def hurtlex_features(text, language):
        text = text.lower()
        text_len = len(text)
        return process(text)/text_len

    language = 'EN'
    conservative_lexicon = read_lexicon('conservative', language)
    inclusive_lexicon = read_lexicon('inclusive', language)

    feat_names = ["ps_conservative", "pa_conservative", "ddf_conservative", "ddp_conservative", "asf_conservative", "pr_conservative", "om_conservative", "qas_conservative", "psps_inclusive", "paps_inclusive", "ddfps_inclusive", "ddpps_inclusive", "asfps_inclusive", "prps_inclusive", "omps_inclusive", "qasps_inclusive"]

    raw_data[feat_names] = raw_data.apply(lambda x: hurtlex_features(x['comment_text_cleaned'], 'en'), axis=1, result_type = 'expand')

    raw_data.to_csv('intermediate_data/hurtlex_features_undersampled.csv', index = False)

    return raw_data, feat_names





# ## Perspective API


def p_api_func(data):
    raw_data = data.copy()
    
    api_key = ''
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +'?key=' + api_key)
    feat_names = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']

    en_attr_dict = {}
    for attr in feat_names:
        en_attr_dict[attr] = {}
    
    def perspective_features(text, language):
        if language == 'en':
            attr_dict = en_attr_dict
            attributes = feat_names
        else:
            attr_dict = es_attr_dict
            attributes = es_attributes
        data_dict = {
            'comment': {'text': text},
            'languages': language,
            'requestedAttributes': attr_dict
        }
        time.sleep(1.01)
        response = requests.post(url=url, data=json.dumps(data_dict)) 
        response_dict = json.loads(response.content)
        pers_dict = {"summary": {}, "span": {}}

        for attr in attributes:
            pers_dict["summary"][attr] = response_dict["attributeScores"][attr]["summaryScore"]["value"]
            curr_span = []
            spanScores = response_dict["attributeScores"][attr]["spanScores"]
            for span in spanScores:
                curr_span.append({'begin': span['begin'], 'end': span['end'], 'score': span['score']['value']})
            pers_dict["span"][attr] = curr_span

        return list(pers_dict['summary'].values())


    raw_data[feat_names] = raw_data.apply(lambda x: perspective_features(x['comment_text_cleaned'], 'en'), axis=1, result_type = 'expand')
    
    raw_data.to_csv('intermediate_data/p_api_features_undersampled.csv', index = False)

    return raw_data, feat_names




# ## BERT


def bert_func(data):
    raw_data = data.copy()
    model = RepresentationModel(
            model_type="bert",
            model_name="bert-base-uncased",
            use_cuda = False
        )

    def get_bert_sent_emb(sentences):
        sentence_vectors = model.encode_sentences(sentences, combine_strategy="mean").tolist()[0]
        return sentence_vectors

    pre = 'feat_'
    feat_names = []
    for i in range(1, 769):
        feat_names.append(pre + str(i))

    raw_data[feat_names] = raw_data.apply(lambda x: get_bert_sent_emb(x['comment_text_cleaned']), axis=1, result_type = 'expand')
    
    raw_data.to_csv('intermediate_data/model_input_data_bert_embedding_final.csv', index = False)

    return raw_data, feat_names




# ## Doc2Vec


def doc_to_vec_func(data):
    raw_data = data.copy()
    
    raw_data['comment_text_cleaned_word_tokenized'] = raw_data['comment_text_cleaned'].apply(lambda x: x.split())
    
    data = list(raw_data['comment_text_cleaned_word_tokenized'])
    
    def tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
    data_training = list(tagged_document(data))

    model = gensim.models.doc2vec.Doc2Vec(vector_size = 20, min_count = 5, epochs = 30)
    
    model.build_vocab(data_training)

    feat_name_pre = 'var_'
    feat_names = [feat_name_pre + str(i) for i in range(1, 21)]

    raw_data[feat_names] = raw_data.apply(lambda x: model.infer_vector(x['comment_text_cleaned_word_tokenized']), axis=1, result_type = 'expand')

    raw_data.to_csv('intermediate_data/doc2vec_features_undersampled.csv', index = False)

    return raw_data, feat_names






# ## Empath


# !pip install empath


def empath_func(data):
    raw_data = data.copy()
    
    lexicon = Empath()

    feat_list_empath = ['toxicity', 'severe_toxicity', 'threat', 'identity_hate', 'violence', 'valuable', 'hate', 'aggression', 'anticipation', 'crime', 'weakness', 'horror', 'swearing_terms', 'kill', 'exasperation', 'body', 'ridicule', 'disgust', 'anger', 'rage']

    def empath_features(text):
        return list(lexicon.analyze(text, categories = feat_list_empath, normalize = True).values())

    raw_data[feat_list_empath] = raw_data.apply(lambda x: empath_features(x['comment_text_cleaned']), axis=1, result_type = 'expand')
    
    raw_data.to_csv('intermediate_data/empath_features_balanced.csv', index = False)

    return raw_data, feat_list_empath


