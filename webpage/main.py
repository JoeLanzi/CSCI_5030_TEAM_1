from flask import Flask, render_template, request
from autocorrect import Autocorrect
import ast

app = Flask(__name__)


autocorrect = Autocorrect()
LANGUAGES = ast.literal_eval(open("language_short_names.txt", "r").read())

#,myfunction=test_func
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


# highlight grammar error, like repeats, & mispelling

@app.route("/home",methods = ["POST","GET"])
def result():
    sentence = request.form['name']
    sentence = sentence.lower()
    autocorrect.language_detect(sentence)
    suggestions = autocorrect.suggestion(sentence.lower())
    newlist = []
    correctedlist = []

    for i in range(len(sentence.split())):
        try:
            if sentence.split()[i] == sentence.split()[i+1]:
                newlist.append('<span class="repeat" style="color: red;"><strike>'+sentence.split()[i]+'</strike></span>')
                continue
            elif ' '.join([sentence.split()[i],sentence.split()[i+1]]) in suggestions[1] and len(suggestions[1][' '.join([sentence.split()[i],sentence.split()[i+1]])])!=0 and sentence.split()[i] not in correctedlist:
                correctedlist.append(sentence.split()[i])
                correctedlist.append(sentence.split()[i+1])
                newlist.append(sentence.split()[i]) # first word
                newlist.append('<span class="err">'+sentence.split()[i+1]+'</span>') # second word


            else:
                if sentence.split()[i] not in correctedlist:
                    newlist.append(sentence.split()[i])
        except IndexError:
            if sentence.split()[i] not in correctedlist:
                newlist.append(sentence.split()[i])
    
    language = LANGUAGES[autocorrect.language]

    correction = []
    for i in correctedlist:
        if len(autocorrect.suggestion(i)[1])!=0:
            correction.append(autocorrect.suggestion(i)[1])

    return render_template("index.html",name = language+': '+' '.join(newlist),sample=sentence,suggestions=correction)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)




'''#%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%#
sentence = 'this is a sample sentence to to see if everuthing is working accordingn to plann'
sentence = sentence.lower()
autocorrect.language_detect(sentence)
suggestions = autocorrect.suggestion(sentence.lower())


print(sentence,':',autocorrect.language,':',suggestions)
#%%
newlist = []
correctedlist = []
for i in range(len(sentence.split())):
    print(i,' ',sentence.split()[i])
    try:
        if sentence.split()[i] == sentence.split()[i+1]:
            newlist.append('<span class="repeat" style="color: red;"><strike>'+sentence.split()[i]+'</strike></span>')
            print('------------- repeat')
            continue
        elif ' '.join([sentence.split()[i],sentence.split()[i+1]]) in suggestions[1] and len(suggestions[1][' '.join([sentence.split()[i],sentence.split()[i+1]])])!=0 and sentence.split()[i] not in correctedlist:
            correctedlist.append(sentence.split()[i+1])

            newlist.append(sentence.split()[i]) # first word
            newlist.append('<span class="err">'+sentence.split()[i+1]+'</span>') # second word
            print("--------- correction")


        else:
            if sentence.split()[i] not in correctedlist:
                newlist.append(sentence.split()[i])


    except IndexError:
        if sentence.split()[i] not in correctedlist:
            newlist.append(sentence.split()[i])

print(correctedlist)

# %%
print(newlist)

# %%

def sample():
    give = []
    for i in correctedlist:
        give.append((i , autocorrect.suggestion(i)[1][i]))
    return(give)

sample()
# %%'''
