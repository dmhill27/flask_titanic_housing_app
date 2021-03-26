from housing_project.model_init_housing import run_housing
from titanic_project.model_init_titanic import run_titanic
from flask import Flask, render_template, url_for, request, send_file
import pandas as pd

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/housing', methods=['GET','POST'])
def housing():
    if request.method == 'POST':
        data = pd.read_csv(request.files.get('input_csv_housing'))
        data = run_housing(data)
        data.to_csv('predictions_housing.csv')
        return render_template('housing.html', data=data.to_html(justify='center', classes='table table-striped table-bordered table-hover table-sm'))
    return render_template('housing.html')

@app.route('/housing/download')
def housing_download():
    return send_file('predictions_housing.csv', attachment_filename='predictions_housing.csv')

@app.route('/titanic', methods=['GET','POST'])
def titanic():
    if request.method == 'POST':
        data = pd.read_csv(request.files.get('input_csv_titanic'))
        data = run_titanic(data)
        data.to_csv('predictions_titanic.csv')
        return render_template('titanic.html', data=data.to_html(justify='center', classes='table table-striped table-bordered table-hover table-sm'))
    return render_template('titanic.html')

@app.route('/titanic/download')
def titanic_download():
    return send_file('predictions_titanic.csv', attachment_filename='predictions_titanic.csv')

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
    app.run(debug=True)