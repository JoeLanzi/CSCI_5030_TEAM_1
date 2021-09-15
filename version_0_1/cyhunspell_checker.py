#pip install cyhunspell

from hunspell import Hunspell

def spell_checker(text):
    s = Hunspell()
    whole = text.split(' ')
    for i in range(len(whole)):
        if whole[i].isalpha() and s.spell(whole[i]) == False:
            whole[i] = '<'+ str(s.suggest(whole[i])[0]) + '>'
        else:
            if whole[i][-1].isalpha()==False and s.spell(whole[i][:-1]) == False:
                whole[i] = '<'+ str(s.suggest(whole[i][:-1])[0]) +'>' + whole[i][-1]
    updated = ' '.join(whole)
    return updated

if __name__ == "__main__":
    text1 = "Dr. Rohde achieved one of nursing's highest honors, that of a Fellowship from the American Academy of Nursing."
    print(spell_checker(text1))
    text2 = "Always active in nursing, serving as falculty, then as profesor, then as Dean of Nursing at SUNY, in Brooklyn, New York, she was Dean and Professor Emeritus of SUNY's Shool of Nursing. Among her other honors was being named as a permanent epresentative to UNICEF, New York, for the Intl. Federation of Business and Professional Woman."
    print(spell_checker(text2)) 
