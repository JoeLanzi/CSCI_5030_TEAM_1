#%% Loading irish data
import pandas as pd
data1 = pd.read_fwf('bible.txt', header=None)
data2 = pd.read_fwf('blogs.txt', header=None)
data3 = pd.read_fwf('legal.txt', header=None)
data4 = pd.read_fwf('news.txt', header=None)
data5 = pd.read_fwf('wiki.txt', header=None)
data = data1[0]+data2[0]+data3[0]+data4[0]+data5[0]
#data = data[0]
# %% Splitting sentence to words
words = []
for i in data:
    for j in str(i).split():
        words.append(j.lower())

#%% remove punctuation & numbers from each words
stripped = str.maketrans('','', '©º�³¬±¼!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~0123456789')
final = [w.translate(stripped) for w in words]

# %% Create dataframe of unique values + frequency
from collections import Counter

df = pd.DataFrame((Counter(final).keys(),Counter(final).values())).T
# %% sort by frequency, delete first row, and reset index
df_clean = df.sort_values(by=[1], ascending=False).iloc[:,:].reset_index(drop=True)
df_clean.drop(df.index[[0,6]], inplace=True)
df_clean.columns=['unique','frequency']
# %%
df_clean[:30]

# %%
df_clean.to_csv('irish_frequency.csv',index=False)
# %%
