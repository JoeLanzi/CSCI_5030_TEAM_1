
#%%
import ast
from langdetect import detect,DetectorFactory
import language_tool_python as lt
from hunspell import Hunspell


LANGUAGES = ast.literal_eval(open("language_short_names.txt", "r").read())

class Autocorrect:
    def __init__(self, language = 'en-US') -> None:
        self.language = language
        self.tool = self.load_dictionary() 
        self._hunspell = Hunspell()
    def language_detect(self,input_string = None) -> str:
        if input_string != None:
            self.input_string = input_string
            DetectorFactory.seed = 0
            self.language = 'en-US' if detect(self.input_string) == 'en' else detect(self.input_string)
            self.tool = self.load_dictionary()
        print(f'Language Detected: {LANGUAGES[self.language]}') 
    
    def load_dictionary(self, language = None):
        language = self.language if language == None else language
        self.language = language
        return lt.LanguageTool(self.language)
        
    def checker(self,input_string):
        self.input_string = input_string
        return self.tool.check(self.input_string)

    def suggestion(self,single_word): # with probability
        if not self._hunspell.spell(single_word):
            return self._hunspell.suggest(single_word)
        elif len(self._hunspell.suffix_suggest(single_word)) != 0:
            return self._hunspell.suffix_suggest(single_word)

    def correct(self,input_string):
        return self.tool.correct(input_string)
