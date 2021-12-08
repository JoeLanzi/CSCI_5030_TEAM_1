#%%
from grammar_checker import Checker
import pandas as pd
from tqdm import tqdm

# %%
checker = Checker('ga')
letters = 'abcdefghilmnoprstuáéíóú-'

#%% with correction500
new = []
data = pd.read_csv ("corrections500.tsv", sep = '\t',header=None,encoding="utf8")
for i in tqdm(data[0][:]):
    if all([characters.lower() in letters for characters in str(i)]):
        new.append(checker.correct(str(i)))
    else:
        new.append(i)
new

#%% accuracy test
from sklearn.metrics import accuracy_score,classification_report
  
test = [str(i) for i in data.iloc[:,1]]
accuracy_score(test,new)

#%%
print(classification_report(test,new))

# %% with input test
data = pd.read_fwf('input-test.txt',header=None)
data

#%% 
data1 = [] 
f = open('input-test.txt', 'r', encoding="utf8")
for i in tqdm(f):
    line = i.rstrip('\n')
    data1.append(line)
data1 = pd.DataFrame(data1)
data1

