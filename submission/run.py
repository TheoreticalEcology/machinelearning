import os
import numpy as np
import pandas as pd
import uuid
import binascii
from sklearn.metrics import accuracy_score, precision_score
from flask import Flask, request, redirect, url_for, flash, render_template, send_file
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/uploads/'

ALLOWED_EXTENSIONS = set(['txt','csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_results(project):
    res_paths = os.listdir("results/"+project.lower()+"/")
    li = []
    for filename in res_paths:
        df = pd.read_csv("results/"+project.lower()+"/"+filename, index_col=0, header=0)
        li.append(df)
    res_df = pd.concat(li, axis=0, ignore_index=True, sort = False)
    #if project != "project5":
    #    res_df.sort_values(by = ["precision"], inplace = True, ascending=False)
    #else:
    res_df.sort_values(by = ["accuracy"], inplace = True, ascending=False)

    return res_df


@app.route('/',  methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      #  if method.validate_on_submit():
            if request.form["projekt"] == "FlowerDownload":
                return redirect(url_for("download_flower"))
            
            if request.form["projekt"] == "DatasetsDownload":
                return redirect(url_for("download_datasets"))

            if request.form['projekt'] == "Results":
                return redirect(url_for('show_Results'))
            else:
                return redirect(url_for('upload', project=request.form['projekt']))
    return '''
    <!doctype html>
    <form method=post>
    <h1>Submit predictions </h1>
    <p>
    <input type="submit" name="projekt" value="Titanic">
    <input type="submit" name="projekt" value="Nasa">
    <input type="submit" name="projekt" value="Wine">
    <input type="submit" name="projekt" value="Flower">
    <input type="submit" name="projekt" value="Results">
    </pr>
    <br>
    <br>
    <h1>Datasets </h1>
    <p>
    <input type="submit" name="projekt" value="DatasetsDownload">
    <input type="submit" name="projekt" value="FlowerDownload">
    </p>
    </form>
    '''
@app.route("/FlowerDownload/", methods=['GET', 'POST'])
def download_flower():
    return send_file("flower.zip",attachment_filename ="flower.zip", as_attachment=True)

@app.route("/DatasetsDownload/", methods=['GET', 'POST'])
def download_datasets():
    return send_file("datasets.RData",attachment_filename ="datasets.RData", as_attachment=True)


@app.route("/show_Results/")
def show_Results():
    titanic =  get_results("titanic").iloc[0:30]
    nasa = get_results("nasa").iloc[0:30]
    wine = get_results("wine").iloc[0:30]
    flower = get_results("flower").iloc[0:30]

    return render_template('show_Results.html', 
                            tables0=[titanic.to_html(classes='data', index = False)], titles0=titanic.columns.values,
                            tables1=[nasa.to_html(classes='data', index = False)], titles1=nasa.columns.values,
                            tables2=[wine.to_html(classes='data', index = False)], titles2=wine.columns.values,
                            tables3=[flower.to_html(classes='data', index = False)], titles3=flower.columns.values
                            )



@app.route('/upload/<project>/', methods=['GET', 'POST'])
def upload(project):
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if project == "Titanic":
                true = pd.read_csv("true/titanic.csv",index_col = 0)
            if project == "Nasa":
                true = pd.read_csv("true/nasa.csv",index_col = 0)
            if project == "Wine":
                true = pd.read_csv("true/wine.csv",index_col = 0)

            if project == "Flower":
                true = pd.read_csv("true/flower.csv",index_col = 0)

            pred = pd.read_csv(file, index_col = 0)
            if true.shape[0] != pred.shape[0]:
                return redirect(url_for('error'))

            if project in ["Titanic", "Nasa"]:
                if pred.iloc[:,0].unique().shape[0] > 2:
                    return redirect(url_for('error'))
            acc = accuracy_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            #prec = precision_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            result = pd.DataFrame(data = {"id": filename, "accuracy": acc}, index = [0])
            result.to_csv("results/"+project.lower()+"/"+filename)

            res_df = get_results(project)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'],"project3", filename))
            return render_template('result.html',
                                    filename = filename, 
                                    acc = acc, 
                                    #prec = prec,  
                                    tables=[res_df.to_html(classes='data')], 
                                    titles=res_df.columns.values)
    return '''
    <!doctype html>
    <title>Projekt 1</title>
    <h1>Submit</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Submit>
    </form>
    '''


@app.route('/error/',  methods=['GET', 'POST'])
def error():
    if request.method == 'POST':
            if request.form['projekt'] == "return":
                return redirect(url_for('index'))   
    return '''
    <!doctype html>
    <h1> Error </h1>
    <p> Predictions must have same length as observed and two columns (index, labels) </>
    <form method=post>
    <input type="submit" name="projekt" value="return">
    </form>
    '''
