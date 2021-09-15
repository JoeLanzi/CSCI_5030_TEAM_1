# Python 3.8.7 64 bit
# py -m pip install langdetect: langdetect 1.0.9 https://pypi.org/project/langdetect/
# py -m pip install cyhunspell: Hunspell 0.5.5   https://pypi.org/project/hunspell/ 
# py -m pip install pyenchant:  pyenchant 3.2.1  https://pypi.org/project/pyenchant/
# Microsoft C++ Build Tools:    https://visualstudio.microsoft.com/visual-cpp-build-tools/


# sequence matcher ??? diflib 
# tokenization ???

# check language -> load dict -> check spelling -> tokenization
#%% Using Language detection
from json import detect_encoding
from random import sample
from langdetect import detect
from langdetect.detector_factory import detect_langs

print(detect("War doesn't show who's right, just who's left."))
print(detect("Ein, zwei, drei, vier"))

# %% Using pyenchant as a spellchecking library
import enchant
#help(enchant)

checker = enchant.Dict("en_US")
print(checker.check("test"))
print(checker.check("onomonipia"))
# %%
from langdetect import detect
import enchant

def spell_sheck(input):
    if detect(input) == 'en':
        test = input.split()
        count = 0
        errors = []
        checker = enchant.Dict("en_US")
        for item in test:
            if checker.check(item) == False:
                errors.append(item)
                test[count] = checker.suggest(item)[0]
            count+=1
        print(" ".join(test))
        return errors 

sample_str = "Hello my namr justlyt is Joe and I live near Chesterfield MO"
spell_sheck(sample_str)

# %%
str = "Hello world"
str.split()
# %%
str = "hellow"
checker = enchant.Dict("en_US")
checker.suggest(str)[0]
# %%
