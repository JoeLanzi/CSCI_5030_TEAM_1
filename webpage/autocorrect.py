# Autocorrect class spelling + grammar
import ast
from langdetect import detect,DetectorFactory
from preprocess import to_n_gram
import language_tool_python as lt
import pickle
from hunspell import Hunspell


LANGUAGES = ast.literal_eval(open("language_short_names.txt", "r").read())

class Autocorrect:
    def __init__(self, language = 'en-US') -> None:
        self.language = language
        self.tool = self.load_dictionary() 
        self._hunspell = Hunspell()
    
    # Detects language
    def language_detect(self,input_string = None, model = "nb") -> str:
        if input_string != None:
            self.input_string = input_string

            # Language Identification using knn
            if model == "knn":
                loaded_model = pickle.load(open('../model_training/models/knn.pickle', 'rb'))
                predict_lang = loaded_model.predict(to_n_gram(self.input_string))[0]
                self.language = [k for k, v in LANGUAGES.items() if v == predict_lang][0]
            
            # Language Identification using langdetect
            elif model == 'ld':
                DetectorFactory.seed = 0
                self.language = 'en-US' if detect(self.input_string) == 'en' else detect(self.input_string)

            # Language Identification using multinb
            else:
                loaded_model = pickle.load(open('../model_training/models/multinb.pickle', 'rb'))
                predict_lang = loaded_model.predict(to_n_gram(self.input_string))[0]
                self.language = [k for k, v in LANGUAGES.items() if v == predict_lang][0]
        
        print("Loading Dictionary")
        self.tool = self.load_dictionary()
        print(f'Language Detected: {LANGUAGES[self.language]}') 
    
    # Loads Dictionary
    def load_dictionary(self, language = None):
        language = self.language if language == None else language
        self.language = language
        return lt.LanguageTool(self.language)
        
    # Check string 
    def checker(self,input_string):
        self.input_string = input_string
        return self.tool.check(self.input_string)

    # word suggession
    def suggestion(self,single_word): # with probability
        if not self._hunspell.spell(single_word):
            return self._hunspell.suggest(single_word)
        elif len(self._hunspell.suffix_suggest(single_word)) != 0:
            return self._hunspell.suffix_suggest(single_word)

    # Output Grammer + Spelling correction
    def correct(self,input_string):
        return self.tool.correct(input_string)
