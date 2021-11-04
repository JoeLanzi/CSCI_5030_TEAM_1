#%% Preprocess Data for Model Prediction
import pickle 
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import normalize

# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
    return None
    
# Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
def to_n_gram(text):
    top1PrecentMixtureVectorizer = pickle.load(open('../model_training/new_models/vectorized_grams', 'rb'))
    text = pd.Series([text]) if type(text) == str else text
    X_top1Percent_test_raw_test = top1PrecentMixtureVectorizer.transform(text)
    top1 = toNumpyArray(normalize(X_top1Percent_test_raw_test, norm='l2', axis=1, copy=True, return_norm=False))
    return top1

# %% Testscd Desktop
#loaded_model = pickle.load(open('../model_training/new_models/multinb.pickle', 'rb'))
#loaded_model.predict(to_n_gram("an lá go mbeidh meáin na Gaeilge agus an Bhéarla ar comhchéim? http://t.co/Fbd9taS via @Twitter slán slán, ag dul chuig rang spin"))[0]

# %%
