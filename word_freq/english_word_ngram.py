#%% Loading corpus
import pandas as pd
from random import shuffle
data = pd.read_fwf('corpus.txt', header=None)
data = data[0]
shuffle(data)
shuffle(data)

# Get word len
words = []
for i in data:
    for j in str(i).split():
        words.append(j.lower())
print(len(words))

# Clean up data -> output to sentence
stripped = str.maketrans('','', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789')
final = []
for i in data[:11000]: #int(len(data)*0.1)
    final.append(' '.join([w.translate(stripped).lower() for w in str(i).split()])) 
final = pd.Series(final, copy=False)
len(final)

#%% Convert to word ngrams
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(ngram_range=(2,2))
tf = vectorizer.fit_transform(final.apply(lambda x: np.str_(x)))
#vectorizer.vocabulary_
#vectorizer.get_feature_names()
#%% Make dictionary of word ngram/frequency
ngram = dict(zip(vectorizer.get_feature_names()  ,tf.toarray().sum(axis=0) ))

#%%
data = pd.DataFrame(ngram.items())
data = data.sort_values(by=[1], ascending=False).reset_index(drop=True)
data.columns=['word ngrams','frequency']
data[:30]

#%%
data['word ngrams'][0]
#%%
data['word ngrams'][0].split()[0]
# %%
len(data)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%
import pickle
y_data = pickle.load(open('../corpus/en_bigram.pickle', 'rb'))

data = pd.DataFrame(y_data.items())
data = data.sort_values(by=[1], ascending=False).reset_index(drop=True)
data.columns=['word ngrams','frequency']
data[:30]
# %%
sample = {i:j for (i,j) in zip(list(y_data.keys()),y_data.values()) if 'of' == i.split()[0]}
#sample= sorted(sample.items(), key=lambda x: x[1], reverse=True)
sample
# %%
for i in list(y_data.keys()):
    print(i)
# %%

# %%
