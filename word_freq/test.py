#%%
from sklearn.feature_extraction.text import CountVectorizer

Doc1 = 'And that was the fallacy. Once I was free to talk with staff members'
Doc2 = 'In the new, stripped-down, every-job-counts business climate, these human'
Doc3 = 'Another reality makes emotional intelligence ever more crucial'
Doc4 = 'The globalization of the workforce puts a particular premium on emotional'
Doc5 = 'As business changes, so do the traits needed to excel. Data tracking'
doc_set = [Doc1, Doc2, Doc3, Doc4, Doc5]

vectorizer = CountVectorizer(ngram_range=(2, 4))
tf = vectorizer.fit_transform(doc_set)
vectorizer.vocabulary_
vectorizer.get_feature_names()
#tf.toarray()

# %%
my_vocabulary= vectorizer.get_feature_names()

vectorizer = CountVectorizer(ngram_range=(2, 3))
vectorizer.fit_transform(my_vocabulary)
tf = vectorizer.transform(doc_set)

vectorizer.vocabulary_
# %%
