from flask import Flask, render_template, request, redirect, session, url_for, Blueprint, jsonify, flash
from flask_login import current_user, login_user, login_required, logout_user, LoginManager

import os

#custom import
from Preprocessing import *


#import routes
from route_preprocessing import preprocessing
from route_classification import classification
from route_regression import regression
from route_clustering import clustering

app = Flask(__name__)

app.register_blueprint(preprocessing)
app.register_blueprint(classification)
app.register_blueprint(regression)
app.register_blueprint(clustering)


@app.route("/", methods = ['POST', 'GET'])
def home():
     return render_template("home.html")

@app.route("/upload", methods = ['POST', 'GET'])
def upload():

    if request.method == 'POST':
        #  def load_dataset():        
        if 'csvFile' not in request.files:
            return "No file part found. Ensure the input name is 'csvFile'.", 400
        
        file = request.files['csvFile']
        file_path = os.path.join(os.getcwd(), "data.csv")
        file.save(file_path)

        if file:
            df, df_column = load_data(file, file_path)

        return render_template("algo.html", df_column = df_column)

    else:
          return render_template("main.html")
    

@app.route("/About-Us", methods = ['POST', 'GET'])
def about():
    return render_template("about.html")


@app.route("/T&S", methods = ['POST', 'GET'])
def ts():
    return render_template("tp.html")


@app.route("/Articles", methods = ['POST', 'GET'])
def articles():
    return render_template("articles.html")


@app.route("/Contact", methods = ['POST', 'GET'])
def contact():
    return render_template("contact.html")


if __name__ == "__main__":                       
        app.run(host='0.0.0.0',
                port=8000,
                debug = True,
                threaded=True)