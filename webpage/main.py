from flask import Flask, render_template, request
from autocorrect import Autocorrect


app = Flask(__name__)


test = Autocorrect()

#,myfunction=test_func
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


# highlight grammar error, like repeats, & mispelling

@app.route("/home",methods = ["POST","GET"])
def result():
    name = request.form['name']
    name = "No errors" if name =='' else name
    name = test.correct(name)
    return render_template("index.html",name = name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)