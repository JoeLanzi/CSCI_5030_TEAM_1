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
        return self.tool.correct(input_string)

# %% Tests

# language_detect
correct = Autocorrect()
correct.language_detect("an lá go mbeidh meáin na Gaeilge agus an Bhéarla ar comhchéim? http://t.co/Fbd9taS via @Twitter slán slán, ag dul chuig rang spin")

# %% load_dictionary
correct = Autocorrect()
correct.load_dictionary('en')
LANGUAGES[correct.language]


