# Python 3.8.7 64 bit
# py -m pip install langdetect: langdetect 1.0.9 https://pypi.org/project/langdetect/
# py -m pip install cyhunspell: Hunspell 0.5.5   https://pypi.org/project/hunspell/ 
# py -m pip install pyenchant:  pyenchant 3.2.1  https://pypi.org/project/pyenchant/
# Microsoft C++ Build Tools:    https://visualstudio.microsoft.com/visual-cpp-build-tools/


# sequence matcher ??? diflib 
# tokenization ???
#%% Using Language detection
from langdetect import detect

print(detect("War doesn't show who's right, just who's left."))
print(detect("Ein, zwei, drei, vier"))

# %% Using pyenchange as a spellchecking library
import enchant
#help(enchant)

checker = enchant.Dict("en_US")
print(checker.check("test"))
print(checker.check("onomonipia"))
# %%
