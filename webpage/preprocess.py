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
    top1PrecentMixtureVectorizer = pickle.load(open('../model_training/models/vectorized_grams', 'rb'))
    text = pd.Series([text]) if type(text) == str else text
    X_top1Percent_test_raw_test = top1PrecentMixtureVectorizer.transform(text)
    top1 = toNumpyArray(normalize(X_top1Percent_test_raw_test, norm='l2', axis=1, copy=True, return_norm=False))
    return top1

# %% Tests
loaded_model = pickle.load(open('../model_training/models/knn.pickle', 'rb'))
loaded_model.predict(to_n_gram("what on earth"))[0]

# %%
