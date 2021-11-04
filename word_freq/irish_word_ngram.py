#%% Loading irish data
import pandas as pd
data1 = pd.read_fwf('bible.txt', header=None)
data2 = pd.read_fwf('blogs.txt', header=None)
data3 = pd.read_fwf('legal.txt', header=None)
data4 = pd.read_fwf('news.txt', header=None)
data5 = pd.read_fwf('wiki.txt', header=None)
data = data1[0]+data2[0]+data3[0]+data4[0]+data5[0]
#%% Clean up data -> output to sentence
stripped = str.maketrans('','', '©º�³¬±¼!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~0123456789')
final = []
for i in data:
    if str(i) != 'nan':
        final.append(' '.join([w.translate(stripped) for w in str(i).split()]))
final = pd.Series(final, copy=False)

# %% Convert to word ngrams
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(ngram_range=(2, 4))
tf = vectorizer.fit_transform(final.apply(lambda x: np.str_(x)))
vectorizer.vocabulary_
vectorizer.get_feature_names()
# %%
