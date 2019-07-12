import os
import numpy as np
import pandas as pd
import uuid
import binascii
from sklearn.metrics import accuracy_score, precision_score
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/submission/uploads/'

ALLOWED_EXTENSIONS = set(['txt','csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_results(project):

    res_paths = os.listdir("results/"+project+"/")
    li = []
    for filename in res_paths:
        df = pd.read_csv("results/"+project+"/"+filename, index_col=0, header=0)
        li.append(df)

    res_df = pd.concat(li, axis=0, ignore_index=True, sort = False)
    if project != "project5":
        res_df.sort_values(by = ["precision"], inplace = True, ascending=False)
    else:
        res_df.sort_values(by = ["accuracy"], inplace = True, ascending=False)
    return res_df


@app.route('/',  methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      #  if method.validate_on_submit():
            if request.form['projekt'] == "liver":
                return redirect(url_for('upload_file1'))
            elif request.form['projekt'] == "exoplanet":
                return redirect(url_for('upload_file2'))
            elif request.form['projekt'] == "Titanic":
                return redirect(url_for('upload_file3'))
            elif request.form['projekt'] == "kickstarter":
                return redirect(url_for('upload_file4'))
            elif request.form['projekt'] == "flower":
                return redirect(url_for('upload_file5'))
            elif request.form['projekt'] == "Results":
                return redirect(url_for('show_Results'))
    return '''
    <!doctype html>
    <form method=post>
    <h1>Submit predictions </h1>
    <p>
    <input type="submit" name="projekt" value="Titanic">
    <input type="submit" name="projekt" value="liver">
    <input type="submit" name="projekt" value="exoplanet">
    <input type="submit" name="projekt" value="kickstarter">
    <input type="submit" name="projekt" value="flower">
    <input type="submit" name="projekt" value="Results">
    </p>
    </form>
    '''

@app.route("/show_Results/")
def show_Results():
    res0 =  get_results("project3").iloc[0:30]
    res1 =  get_results("project1").iloc[0:30]
    res2 =  get_results("project2").iloc[0:30]
    res3 =  get_results("project4").iloc[0:30]
    res4 =  get_results("project5").iloc[0:30]
    return render_template('show_Results.html', 
                            tables0=[res0.to_html(classes='data', index = False)], titles0=res0.columns.values,
                            tables1=[res1.to_html(classes='data', index = False)], titles1=res1.columns.values,  
                            tables2=[res2.to_html(classes='data', index = False)], titles2=res2.columns.values,
                            tables3=[res3.to_html(classes='data', index = False)], titles3=res3.columns.values,
			            tables4=[res4.to_html(classes='data', index = False)], titles4=res4.columns.values)





@app.route('/projekt1/', methods=['GET', 'POST'])
def upload_file1():
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
            true = pd.read_csv("true/project_1.true.csv",index_col = 0)
            pred = pd.read_csv(file, index_col = 0)
            if true.shape[0] != pred.shape[0]:
                return redirect(url_for('error'))
            if pred.iloc[:,0].unique().shape[0] > 2:
                 return redirect(url_for('error'))
            acc = accuracy_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            prec = precision_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"project1", filename))
            result = pd.DataFrame(data = {"id": filename, "accuracy": acc, "precision": prec}, index = [0])
            result.to_csv("results/project1/"+filename)
            res_df = get_results("project1")
            return render_template('result.html',filename = filename, acc = acc, prec = prec,  tables=[res_df.to_html(classes='data')], titles=res_df.columns.values)
    return '''
    <!doctype html>
    <title>Projekt 2</title>
    <h1>Submit</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Submit>
    </form>
    '''



@app.route('/projekt2/', methods=['GET', 'POST'])
def upload_file2():
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
            true = pd.read_csv("true/project_2.true.csv",index_col = 0)
            pred = pd.read_csv(file, index_col = 0)
            if true.shape[0] != pred.shape[0]:
                return redirect(url_for('error'))
            if pred.iloc[:,0].unique().shape[0] > 2:
                 return redirect(url_for('error'))
            acc = accuracy_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            prec = precision_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            result = pd.DataFrame(data = {"id": filename, "accuracy": acc, "precision": prec}, index = [0])
            result.to_csv("results/project2/"+filename)
            res_df = get_results("project2")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"project2", filename))
            return render_template('result.html',filename = filename, acc = acc, prec = prec,  tables=[res_df.to_html(classes='data')], titles=res_df.columns.values)
    return '''
    <!doctype html>
    <title>Projekt 1</title>
    <h1>Submit</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Submit>
    </form>
    '''


@app.route('/titanic/', methods=['GET', 'POST'])
def upload_file3():
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
            true = pd.read_csv("true/titanic.csv",index_col = 0)
            pred = pd.read_csv(file, index_col = 0)
            if true.shape[0] != pred.shape[0]:
                return redirect(url_for('error'))
            if pred.iloc[:,0].unique().shape[0] > 2:
                 return redirect(url_for('error'))
            acc = accuracy_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            prec = precision_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            result = pd.DataFrame(data = {"id": filename, "accuracy": acc, "precision": prec}, index = [0])
            result.to_csv("results/project3/"+filename)
            res_df = get_results("project3")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"project3", filename))
            return render_template('result.html',filename = filename, acc = acc, prec = prec,  tables=[res_df.to_html(classes='data')], titles=res_df.columns.values)
    return '''
    <!doctype html>
    <title>Projekt 1</title>
    <h1>Submit</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Submit>
    </form>
    '''


@app.route('/kickstarter/', methods=['GET', 'POST'])
def upload_file4():
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
            true = pd.read_csv("true/kickstarter.csv",index_col = 0)
            pred = pd.read_csv(file, index_col = 0)
            if true.shape[0] != pred.shape[0]:
                return redirect(url_for('error'))
            if pred.iloc[:,0].unique().shape[0] > 2:
                 return redirect(url_for('error'))
            acc = accuracy_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            prec = precision_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            result = pd.DataFrame(data = {"id": filename, "accuracy": acc, "precision": prec}, index = [0])
            result.to_csv("results/project4/"+filename)
            res_df = get_results("project4")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"project4", filename))
            return render_template('result.html',filename = filename, acc = acc, prec = prec,  tables=[res_df.to_html(classes='data')], titles=res_df.columns.values)
    return '''
    <!doctype html>
    <title>Projekt 1</title>
    <h1>Submit</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Submit>
    </form>
    '''

@app.route('/flower/', methods=['GET', 'POST'])
def upload_file5():
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
            true = pd.read_csv("true/flower.csv",index_col = 0)
            pred = pd.read_csv(file, index_col = 0)
            if true.shape[0] != pred.shape[0]:
                return redirect(url_for('error'))
            if pred.iloc[:,0].unique().shape[0] > 5:
                return redirect(url_for('error'))
            acc = accuracy_score(true.iloc[:,0].values.astype(int), pred.iloc[:,0].values.astype(int))
            result = pd.DataFrame(data = {"id": filename, "accuracy": acc}, index = [0])
            result.to_csv("results/project5/"+filename)
            res_df = get_results("project5")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"project5", filename))
            return render_template('result.html',filename = filename, acc = acc, prec = 0.0,  tables=[res_df.to_html(classes='data')], titles=res_df.columns.values)
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
