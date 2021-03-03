from flask import Flask, render_template, url_for, request, send_file
from housing_project.housing_model_init import housing_run
#from housing_project import housing_run
#from housing_project import *
import pandas as pd

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/housing', methods=['GET','POST'])
def housing():
    if request.method == 'POST':
        data = pd.read_csv(request.files.get('input_csv'))
        data = housing_run(data)
        data.to_csv('housing_predictions.csv')
        return render_template('housing.html', data=data.to_html(justify='left', classes='table table-striped table-bordered table-hover table-sm'))
    return render_template('housing.html')

@app.route('/housing/download')
def housing_download():
    return send_file('housing_predictions.csv', attachment_filename='housing_predictions.csv')
'''
@app.route('/housing', methods=['GET','POST'])
def housing():
    if request.method == 'POST':
        data = pd.read_csv(request.files.get('input_csv'))
        data = housing_run(data)
        return render_template('result.html', data=data.to_html(justify='left', classes='table table-striped table-bordered table-hover table-sm'))
    return render_template('housing.html')

@app.route('/result', methods=['POST'])
def result():
    data = pd.read_csv(request.files.get('input_csv'))
    data = housing_run(data)
    return render_template('result.html', data=data.to_html(justify='left', classes='table table-striped table-bordered table-hover table-sm'))
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)