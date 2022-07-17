import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")



def performance_metrics_creation(y_train, y_train_pred, y_test, y_test_pred, result, feat_extr_method, model_name):
    # precision_score
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)

    # recall_score
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)

    # f1_score
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # accuracy_score
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    temp_df = pd.DataFrame({
        'feature_extraction_method' : [feat_extr_method],
        'model' : [model_name],
        'train_precision' : [train_precision],
        'test_precision' : [test_precision],
        'train_recall' : [train_recall],
        'test_recall' : [test_recall],
        'train_f1' : [train_f1],
        'test_f1' : [test_f1],
        'train_accuuracy' : [train_accuracy],
        'test_accuracy' : [test_accuracy]
    })

    result = pd.concat([result, temp_df], axis = 0)

    return result



## Support Vector Classifier

def get_prediction_svc(X_train, X_test, y_train, y_test):
    print("====================================Support Vector Classifier=====================================")
    clf_svc = SVC(kernel='rbf', random_state = 53)
    print('svc object created')

    clf_svc.fit(X_train, y_train)
    print('svc object fitted')

    pred_train_svc = clf_svc.predict(X_train)
    print('train prediction done')

    pred_test_svc = clf_svc.predict(X_test)
    print('test prediction done')
    
    return pred_train_svc, pred_test_svc



## Random Forest Classifier

def get_prediction_rfc(X_train, X_test, y_train, y_test):
    print("====================================Random Forest Classifier=====================================")
    clf_rfc = RandomForestClassifier()
#     clf_svc = SVC(kernel='rbf', random_state = 53)
    print('random forest classifier object created')

    clf_rfc.fit(X_train, y_train)
    print('random forest classifier object fitted')

    pred_train_rfc = clf_rfc.predict(X_train)
    print('train prediction done')

    pred_test_rfc = clf_rfc.predict(X_test)
    print('test prediction done')
    
    return pred_train_rfc, pred_test_rfc



## X G Boost Classifier

def get_prediction_xgb(X_train, X_test, y_train, y_test):
    print("====================================XGBoost Classifier=====================================")
    clf_xgb = XGBClassifier()
    print('X G Boost classifier object created')

    clf_xgb.fit(X_train, y_train)
    print('X G Boost classifier object fitted')

    pred_train_xgb = clf_xgb.predict(X_train)
    print('train prediction done')

    pred_test_xgb = clf_xgb.predict(X_test)
    print('test prediction done')
    
    return pred_train_xgb, pred_test_xgb



## Artificial Neural Network

def get_prediction_ann(X_train, X_test, y_train, y_test):
    print("====================================Artificial Neural Network=====================================")
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)

    NN_model = Sequential()

    NN_model.add(Dense(X_train.shape[0], kernel_initializer='normal', input_dim = X_train.shape[1], activation='relu'))

    NN_model.add(Dense(512, activation='relu'))
    NN_model.add(Dense(1024, activation='relu'))
    NN_model.add(Dropout(0.5))

    NN_model.add(Dense(1, activation='sigmoid'))

    NN_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

    NN_model.fit(X_train_array, y_train, epochs = 10)

    y_train_pred_arr = NN_model.predict(X_train_array)
    y_test_pred_arr = NN_model.predict(X_test_array)

    y_train_pred = [1 if i[0] >= 0.5 else 0 for i in y_train_pred_arr]
    y_test_pred = [1 if i[0] >= 0.5 else 0 for i in y_test_pred_arr]

    return y_train_pred, y_test_pred



## LSTM

def get_prediction_lstm(X_train, X_test, y_train, y_test):
    print("====================================LSTM=====================================")
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)
    X_train_array.shape, X_test_array.shape

    X_train_reshaped = np.reshape(X_train_array, (X_train_array.shape[0], 1, X_train_array.shape[1]))
    X_test_reshaped = np.reshape(X_test_array, (X_test_array.shape[0], 1, X_test_array.shape[1]))
    X_train_reshaped.shape, X_test_reshaped.shape

    model_lstm = Sequential()
    model_lstm.add(LSTM(32, input_shape=(1, X_train_reshaped.shape[2]), activation='tanh'))
    model_lstm.add(Dense(128, activation='relu'))
    model_lstm.add(Dense(256, activation='relu'))
    model_lstm.add(Dense(128, activation='relu'))
    model_lstm.add(Dense(600, activation='relu'))
    model_lstm.add(Dense(1000, activation='relu'))
    model_lstm.add(Dense(1, activation='sigmoid'))

    model_lstm.compile(loss='BinaryCrossentropy', optimizer='Adam')

    model_lstm.fit(X_train_reshaped, y_train, epochs = 10)

    y_train_pred_arr_lstm = model_lstm.predict(X_train_reshaped)
    y_test_pred_arr_lstm = model_lstm.predict(X_test_reshaped)

    y_train_pred_lstm = [1 if i[0] >= 0.5 else 0 for i in y_train_pred_arr_lstm]
    y_test_pred_lstm = [1 if i[0] >= 0.5 else 0 for i in y_test_pred_arr_lstm]
    
    return y_train_pred_lstm, y_test_pred_lstm




## Bidirectional LSTM

def get_prediction_bi_lstm(X_train, X_test, y_train, y_test):
    print("====================================Bidirectional LSTM=====================================")
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)
    X_train_array.shape, X_test_array.shape

    X_train_reshaped = np.reshape(X_train_array, (X_train_array.shape[0], 1, X_train_array.shape[1]))
    X_test_reshaped = np.reshape(X_test_array, (X_test_array.shape[0], 1, X_test_array.shape[1]))
    X_train_reshaped.shape, X_test_reshaped.shape

    model_bi_lstm = Sequential()
    model_bi_lstm.add(Bidirectional(LSTM(32, input_shape=(1, X_train_reshaped.shape[2]), activation='tanh')))
    model_bi_lstm.add(Dense(128, activation='relu'))
    model_bi_lstm.add(Dense(256, activation='relu'))
    model_bi_lstm.add(Dense(128, activation='relu'))
    model_bi_lstm.add(Dense(600, activation='relu'))
    model_bi_lstm.add(Dense(1000, activation='relu'))
    model_bi_lstm.add(Dense(1, activation='sigmoid'))

    model_bi_lstm.compile(loss='BinaryCrossentropy', optimizer='Adam')

    model_bi_lstm.fit(X_train_reshaped, y_train, epochs = 10)

    y_train_pred_arr_bi_lstm = model_bi_lstm.predict(X_train_reshaped)
    y_test_pred_arr_bi_lstm = model_bi_lstm.predict(X_test_reshaped)

    y_train_pred_bi_lstm = [1 if i[0] >= 0.5 else 0 for i in y_train_pred_arr_bi_lstm]
    y_test_pred_bi_lstm = [1 if i[0] >= 0.5 else 0 for i in y_test_pred_arr_bi_lstm]
    
    return y_train_pred_bi_lstm, y_test_pred_bi_lstm



def main_function(X_train, X_test, y_train, y_test, feat_extr_method):
    result = pd.DataFrame({
        'feature_extraction_method' : [],
        'model' : [],
        'train_precision' : [],
        'test_precision' : [],
        'train_recall' : [],
        'test_recall' : [],
        'train_f1' : [],
        'test_f1' : [],
        'train_accuuracy' : [],
        'test_accuracy' : []
    })
    
#     pred_train_svc, pred_test_svc = get_prediction_svc(X_train, X_test, y_train, y_test)
    
#     result = performance_metrics_creation(y_train, pred_train_svc, y_test, pred_test_svc, result, feat_extr_method, 'Vanilla Support Vector Classifier')
    
    
    
    pred_train_rfc, pred_test_rfc = get_prediction_rfc(X_train, X_test, y_train, y_test)
    
    result = performance_metrics_creation(y_train, pred_train_rfc, y_test, pred_test_rfc, result, feat_extr_method, 'Vanilla Random Forest Classifier')
    
    
    
    pred_train_xgb, pred_test_xgb = get_prediction_xgb(X_train, X_test, y_train, y_test)
    
    result = performance_metrics_creation(y_train, pred_train_xgb, y_test, pred_test_xgb, result, feat_extr_method, 'Vanilla XG Boost Classifier')
    
    
    
#     pred_train_ann, pred_test_ann = get_prediction_ann(X_train, X_test, y_train, y_test)
    
#     result = performance_metrics_creation(y_train, pred_train_ann, y_test, pred_test_ann, result, feat_extr_method, 'Basic ANN')
    
    
    
    pred_train_lstm, pred_test_lstm = get_prediction_lstm(X_train, X_test, y_train, y_test)
    
    result = performance_metrics_creation(y_train, pred_train_lstm, y_test, pred_test_lstm, result, feat_extr_method, 'Basic LSTM')
    
    
    
    pred_train_bi_lstm, pred_test_bi_lstm = get_prediction_bi_lstm(X_train, X_test, y_train, y_test)
    
    result = performance_metrics_creation(y_train, pred_train_bi_lstm, y_test, pred_test_bi_lstm, result, feat_extr_method, 'Basic Bidirectional LSTM')
    
    
    
    return result

