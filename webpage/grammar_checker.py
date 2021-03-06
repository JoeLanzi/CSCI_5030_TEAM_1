#%% IMPORT DATA
import pickle
import itertools
from collections import Counter
import math

class Checker():
    def __init__(self, language = 'en-US') -> None:
        self.language = language
        self.DICTIONARY,self.BIGRAMS, self.LETTERS = [],[],[]
        self.repeated_words, self.correct_grammar = [],[]
        self.load_dictionary()

    # Load Dictionary based on language
    def load_dictionary(self,language = None):
        language = self.language if language == None else language
        if language in ['en','en-US']:
            self.DICTIONARY = pickle.load(open('../corpus/en_unigram.pickle', 'rb'))
            self.BIGRAMS = pickle.load(open('../corpus/en_bigram.pickle', 'rb'))
            self.LETTERS = 'abcdefghijklmnopqrstuvwxyz' 
        elif language in ['ga','ga-IE']:
            self.DICTIONARY = pickle.load(open('../corpus/irish_unigram.pickle', 'rb'))
            self.BIGRAMS = pickle.load(open('../corpus/irish_bigram.pickle', 'rb'))
            self.LETTERS = 'abcdefghilmnoprstuáéíóú'

    # Probability of word from dictionary
    def P(self, word, dictionary = None): 
        dictionary = self.DICTIONARY if dictionary == None else dictionary
        return dictionary[word]/sum(dictionary.values()) if word in dictionary else 0

    # return most probable word
    def correction(self,word):
        return max(self.candidates(word), key=self.P)

    # one edit for words of up to four characters 
    # two edits for up to twelve characters
    # three for longer words
    # generate possible spelling correciton for word
    def candidates(self,word):
        if len(word)<5:
            return self.known([word]) | self.known(self.edits1(word)) or self.known(self.edits2(word)) or set([word])
        elif len(word)>=5 and len(word)<13:
            return self.known([word]) | self.known(self.edits1(word)) or self.known(self.edits2(word))  or set([word])
        else:
            return self.known([word]) | self.known(self.edits1(word)) | self.known(self.edits2(word)) or set([word])

    # in dictionary
    def known(self,words): 
        return set(w for w in words if w in self.DICTIONARY)

    # one edit away for word
    def edits1(self,word,delete=True):
        letters    =  self.LETTERS                      
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        
        if delete:
            deletes    = [L + R[1:]               for L, R in splits if R]
        else:
            deletes    = ['']
        return set(deletes + transposes + replaces + inserts)

    # two edits away for word
    def edits2(self,word,delete=True):
        new = self.edits1(word,delete)                             # Get list of all the one edits
        [new.update(self.edits1(i)) for i in self.edits1(word)]    # Iterate through all the objects in one edit list
        return new 

    # three edits away for word
    def edits3(self,word,delete=True):
        new = self.edits2(word,delete)
        [new.update(self.edits1(i)) for i in self.edits2(word)]
        return new

    # Probability from BIGRAMS Dictionary
    def check(self,bigram):
        if len(bigram.split()) < 2:
            return self.candidates(bigram)
        else:
            second_word_candicate = list(itertools.product([bigram.split()[0]], list(self.candidates(bigram.split()[1]))))
            possible = [' '.join(list(i)) for i in second_word_candicate]
            prob = dict(zip(possible,[self.P(i,self.BIGRAMS) for i in possible])) #P[word|word[0]] = P[word[1]] 
            return {x:y for x,y in dict(Counter(prob).most_common(50)).items() if y != 0} if possible != {} else bigram

    # strike words
    def strike(self,text):
        result = ''
        for c in text:
            result = result + c + '\u0336'
        return result

    # collects grammar corrections and repeated words
    def tool(self,sentence):
        split = sentence.split()
        if not split[-1][-1].isalpha(): #considering the punctuation at the end of the sentence
            split[-1] = split[-1][:-1]
        corrected = False
        repeated_words = []
        correct_grammar = {}

        # for single words
        if len(split) < 2:
            checked = list(self.check(sentence))
            new_check = checked[:math.ceil(len(checked)*0.9)]
            if(sentence not in [j for j in new_check]): 
                correct_grammar[sentence] = self.candidates(sentence)

            self.repeated_words = repeated_words
            self.correct_grammar = correct_grammar
            return self.correction(sentence)
            
        # for sentence
        else:
            for i in range(len(split)-1):
                bigram = ' '.join([split[i],split[i+1]])
                checked = self.check(bigram)
                new_check = list(checked.keys())[:math.ceil(len(checked)*0.9)]
                
                if(split[i]==split[i+1]):
                    repeated_words.append(split[i+1])

                elif(split[i+1] not in [j.split()[1] for j in new_check]): 
                    correct_grammar[bigram] = [k for k in list(new_check)[:10]]
                
            self.repeated_words = repeated_words
            self.correct_grammar = correct_grammar


    # delete word from sentence
    def delete(self,sentence,word):
        return sentence.replace(' '+word,'',1)

    # replace word with replacement word
    def replace(self,sentence,word,replacement):
        return sentence.replace(word,replacement,1)

# %% TESTS
'''
checker = Checker('en')

sample_sentence = "this might be the the end of this sentece"
checker.tool(sample_sentence)

print('Original: ', sample_sentence)
print('Repeated words: ', checker.repeated_words)
print('Grammar suggestions: ', checker.correct_grammar)
sample_sentence = checker.delete(sample_sentence,checker.repeated_words[-1])
print('Delete repeats: ', sample_sentence)

sample_sentence = checker.replace(sample_sentence,list(checker.correct_grammar.keys())[0],list(checker.correct_grammar.values())[0][0])
print('Replace grammar: ', sample_sentence)
'''
