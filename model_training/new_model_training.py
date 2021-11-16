#%% Load Data
# Loading english data
import pandas as pd
data0 = pd.read_fwf('../word_freq/corpus.txt', header=None)
english = data0[0]

# Loading irish data
data1 = pd.read_fwf('../word_freq/bible.txt', header=None)
data2 = pd.read_fwf('../word_freq/blogs.txt', header=None)
data3 = pd.read_fwf('../word_freq/legal.txt', header=None)
data4 = pd.read_fwf('../word_freq/news.txt', header=None)
data5 = pd.read_fwf('../word_freq/wiki.txt', header=None)
irish = data1[0]+data2[0]+data3[0]+data4[0]+data5[0]
# %%Clean up data -> output to Dataframe
en_stripped = str.maketrans('','', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789')
ga_stripped = str.maketrans('','', '©º�³¬±¼!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~0123456789')

en_final = []
ga_final = []
for i in english:
    if str(i).lower() != 'nan':
        en_final.append(' '.join([w.translate(en_stripped) for w in str(i).split()]))
for i in irish:
    if str(i).lower() != 'nan':
        ga_final.append(' '.join([w.translate(ga_stripped) for w in str(i).split()]))

en_final = pd.DataFrame(en_final, copy=False,columns=['text'])
en_final['language'] = 'English'
ga_final = pd.DataFrame(ga_final, copy=False,columns=['text'])
ga_final['language'] = 'Irish'

# Final Data
data = pd.concat([en_final,ga_final])
languages = set(data['language'])
#%% Shuffle & Train Test Split
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data = shuffle(data)

X=data['text']
y=data['language']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#%% Aggregate Unigrams per language
def train_lang_dict(X_raw_counts, y_train):
    lang_dict = {}
    for i in range(len(y_train)):
        lang = y_train[i]
        v = np.array(X_raw_counts[i])
        if not lang in lang_dict:
            lang_dict[lang] = v
        else:
            lang_dict[lang] += v
            
    # to relative
    for lang in lang_dict:
        v = lang_dict[lang]
        lang_dict[lang] = v / np.sum(v)
        
    return lang_dict


#%% Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
from sklearn.feature_extraction.text import CountVectorizer

top1PrecentMixtureVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), min_df=1e-2)
X_top1Percent_train_raw = top1PrecentMixtureVectorizer.fit_transform(X_train)
X_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(X_test)

language_dict_top1Percent = train_lang_dict(X_top1Percent_train_raw.toarray(), y_train.values)

top1PercentFeatures = top1PrecentMixtureVectorizer.get_feature_names()
print('Length of features', len(top1PercentFeatures))
print('')

# get relevant 50 n-grams for each languages
def getRelevantGramsPerLanguage(features, language_dict, top=50):
    relevantGramsPerLanguage = {}
    for lang in languages:
        chars = []
        relevantGramsPerLanguage[lang] = chars
        v = language_dict[lang]
        sortIndex = (-v).argsort()[:top]
        for i in range(len(sortIndex)):
            chars.append(features[sortIndex[i]])
    return relevantGramsPerLanguage

top50PerLanguage_dict = getRelevantGramsPerLanguage(top1PercentFeatures, language_dict_top1Percent)

# top50
allTop50 = []
for lang in top50PerLanguage_dict:
    allTop50 += set(top50PerLanguage_dict[lang])

top50 = list(set(allTop50))
    
print('All items:', len(allTop50))
print('Unique items:', len(top50))



# %% define getRelevantColumnIndices for top 50 n-grams
def getRelevantColumnIndices(allFeatures, selectedFeatures):
    relevantColumns = []
    for feature in selectedFeatures:
        relevantColumns = np.append(relevantColumns, np.where(allFeatures==feature))
    return relevantColumns.astype(int)

relevantColumnIndices = getRelevantColumnIndices(np.array(top1PercentFeatures), top50)


X_top50_train_raw = np.array(X_top1Percent_train_raw.toarray()[:,relevantColumnIndices])
X_top50_test_raw = X_top1Percent_test_raw.toarray()[:,relevantColumnIndices] 

print('train shape', X_top50_train_raw.shape)
print('test shape', X_top50_test_raw.shape)
# %% Define some functions for multinominal Naive Bays

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import scipy

# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
    print(data_type)
    return None


def normalizeData(train, test):
    train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)
    test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)
    return train_result, test_result

def applyNaiveBayes(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def plot_F_Scores(y_test, y_predict):
    f1_micro = f1_score(y_test, y_predict, average='micro')
    f1_macro = f1_score(y_test, y_predict, average='macro')
    f1_weighted = f1_score(y_test, y_predict, average='weighted')
    print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))

def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
    allLabels = list(set(list(y_test) + list(y_predict)))
    allLabels.sort()
    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
    unqiueLabel = np.unique(allLabels)
    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    sn.set(font_scale=0.8) # for label size
    sn.set(rc={'figure.figsize':(15, 15)})
    sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')# font size
    plt.show()


#%% MultiNB
# Top 1%
X_top1Percent_train, X_top1Percent_test = normalizeData(X_top1Percent_train_raw, X_top1Percent_test_raw)
y_predict_nb_top1Percent = applyNaiveBayes(X_top1Percent_train, y_train, X_top1Percent_test)
plot_F_Scores(y_test, y_predict_nb_top1Percent)
plot_Confusion_Matrix(y_test, y_predict_nb_top1Percent, "Reds")

# %% Applying Multinominal NB to Top 50 n-grams
X_top50_train, X_top50_test = normalizeData(X_top50_train_raw, X_top50_test_raw)
y_predict_nb_top50 = applyNaiveBayes(X_top50_train, y_train, X_top50_test)
plot_F_Scores(y_test, y_predict_nb_top50)
plot_Confusion_Matrix(y_test, y_predict_nb_top50, "Greens")

# %% Training NB Model w/ top 1% n-grams to Save
import pickle
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_top1Percent_train, y_train)

# save the model to disk
modelname = './new_models/multinb.pickle'
pickle.dump(model, open(modelname, 'wb'))
# %% save vectorized ngrams

vectorized = './new_models/vectorized_grams'
pickle.dump(top1PrecentMixtureVectorizer, open(vectorized, 'wb'))
# %%
