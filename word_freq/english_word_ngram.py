#%% Loading corpus
import pandas as pd
data = pd.read_fwf('corpus.txt', header=None)
data = data[0]

#%% Clean up data -> output to sentence
stripped = str.maketrans('','', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789')
final = []
for i in data:
    final.append(' '.join([w.translate(stripped) for w in str(i).split()]))
    ' '.join([w.translate(stripped) for w in str(i).split()])
final = pd.Series(final, copy=False)

# %% Convert to word ngrams
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(ngram_range=(2, 4))
tf = vectorizer.fit_transform(final.apply(lambda x: np.str_(x)))
vectorizer.vocabulary_
vectorizer.get_feature_names()
# %%
