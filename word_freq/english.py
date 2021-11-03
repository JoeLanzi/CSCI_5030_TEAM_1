#%% Loading corpus
import pandas as pd
data = pd.read_fwf('corpus.txt', header=None)
data = data[0]
# %% Splitting sentence to words
words = []
for i in data:
    for j in str(i).split():
        words.append(j.lower())

#%% remove punctuation & numbers from each words
stripped = str.maketrans('','', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789')
final = [w.translate(stripped) for w in words]

# %% Create dataframe of unique values + frequency
from collections import Counter

df = pd.DataFrame((Counter(final).keys(),Counter(final).values())).T
# %% sort by frequency, delete first row, and reset index
df_clean = df.sort_values(by=[1], ascending=False).reset_index(drop=True)
df_clean.drop(df.index[[0]], inplace=True)
df_clean.columns=['unique','frequency']
# %%
df_clean[:30]

# %%
df_clean.to_csv('english_frequency.csv',index=False)
# %%
