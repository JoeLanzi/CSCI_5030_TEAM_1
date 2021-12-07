#%%
import ast
from preprocess import to_n_gram
from grammar_checker import Checker
import pickle


LANGUAGES = ast.literal_eval(open("language_short_names.txt", "r").read())

class Autocorrect:
    def __init__(self, language = 'en-US') -> None:
        self.language = language
        self.tool = self.load_dictionary() 
    
    # Detects language
    def language_detect(self,input_string = None) -> str:
        if input_string != None:
            self.input_string = input_string

            # Language Identification using multinb
            loaded_model = pickle.load(open('../model_training/new_models/multinb.pickle', 'rb'))
            predict_lang = loaded_model.predict(to_n_gram(self.input_string))[0]
            self.language = [k for k, v in LANGUAGES.items() if v == predict_lang][0]
        
        print("Loading Dictionary")
        self.tool = self.load_dictionary()
        print(f'Language Detected: {LANGUAGES[self.language]}') 
    
    # Loads Dictionary
    def load_dictionary(self, language = None):
        language = self.language if language == None else language
        self.language = language
        return Checker(self.language)
        
    # word suggession
    def suggestion(self,input_string): # with probability
        self.tool.tool(input_string)
        return [self.tool.repeated_words,self.tool.correct_grammar]


    # Output Grammer + Spelling correction
    def correct(self,input_string):
        #return self.tool.correct(input_string)
        pass

# %% Tests
'''
correct = Autocorrect()
sentence = "an lá go mbeidh meáin na Gaeilge agus an Bhéarla ar comhchéim"
correct.language_detect(sentence.lower()) 
correct.suggestion(sentence.lower())
'''
#%% Spell check for html
'''
sentence = "this is a a sample sentece"
correct = Autocorrect()
correct.language_detect(sentence)
samplelist = correct.suggestion(sentence)
# %%
newlist = []
corrected = False
for i in range(len(sentence.split())):
    try:
        if sentence.split()[i] == sentence.split()[i+1]:
            newlist.append('<div class="err">'+sentence.split()[i]+'</div>')
            continue
        elif ' '.join([sentence.split()[i],sentence.split()[i+1]]) in samplelist[1]:
            newlist.append('<div class="err">'+' '.join([sentence.split()[i],sentence.split()[i+1]])+'</div>')
            corrected = True
            continue
        newlist.append(sentence.split()[i])
    except IndexError:
        if not corrected:
            newlist.append(sentence.split()[i])
        else:
            pass
' '.join(newlist)
'''

