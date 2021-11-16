from flask import Flask, render_template, request
from autocorrect import Autocorrect


app = Flask(__name__)


autocorrect = Autocorrect()

#,myfunction=test_func
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


# highlight grammar error, like repeats, & mispelling

@app.route("/home",methods = ["POST","GET"])
def result():
    sentence = request.form['name']
    autocorrect.language_detect(sentence)
    suggestions = autocorrect.suggestion(sentence.lower())
    newlist = []
    corrected = False
    for i in range(len(sentence.split())):
        try:
            if sentence.split()[i] == sentence.split()[i+1]:
                newlist.append('<span style="color: red;"><strike>'+sentence.split()[i]+'</strike></span>')
                continue
            elif ' '.join([sentence.split()[i],sentence.split()[i+1]]) in suggestions[1]:
                newlist.append(sentence.split()[i])
                newlist.append('<span class="err">'+sentence.split()[i+1]+'</span>')
                corrected = True
                continue
            newlist.append(sentence.split()[i])
        except IndexError:
            if not corrected:
                newlist.append(sentence.split()[i])
            else:
                pass
    
    return render_template("index.html",name = ' '.join(newlist))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
