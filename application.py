
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


application = app = Flask(__name__,static_url_path='',static_folder="static",template_folder='templates')
#app = Flask(__name__, static_folder="your path to static")

modelRandomForest = pickle.load(open('Pickles/RandomForestRegression/model.pkl', 'rb'))
modelLinearRegression = pickle.load(open('Pickles/LinearRegression/model.pkl', 'rb'))
modelDecisiontreeClassification = pickle.load(open('Pickles/DecisiontreeClassification/model.pkl', 'rb'))
modelLogisticRegression = pickle.load(open('Pickles/LogisticRegression/model.pkl', 'rb'))

#generic pages start
@app.route('/')
def home():
    return render_template('/Homepage/home.html')

@app.route('/about')
def about():
    return render_template('/Homepage/about.html')

@app.route('/blog')
def blog():
    return render_template('/Homepage/blog.html')
#generic pages ends


# Project list starts 
@app.route('/projectlistregression')
def projectlistregression():
    return render_template('/Input/projectlistregression.html')

@app.route('/projectlistclassification')
def projectlistclassification():
    return render_template('/Input/projectlistclassification.html')

@app.route('/projectlistclustering')
def projectlistclustering():
    return render_template('/Input/projectlistclustering.html')


@app.route('/projectlistneuralnetwork')
def projectlistneuralnetwork():
    return render_template('/Input/projectlistneuralnetwork.html')

@app.route('/projectlisttimeseries')
def projectlisttimeseries():
    return render_template('/Input/projectlisttimeseries.html')


@app.route('/projectlistnlp')
def projectlistnlp():
    return render_template('/Input/projectlistnlp.html')


@app.route('/projectlistrnn')
def projectlistrnn():
    return render_template('/Input/projectlistrnn.html')

# Project list ends

#Input form starts
@app.route('/randomforestregressionform')
def randomforestregressionform():
    return render_template('/Input/randomforest-regression-form.html')

@app.route('/linearregressionform')
def linearregressionform():
    return render_template('/Input/linear-regression-form.html')

@app.route('/logisticregressionform')
def logisticregressionform():
    return render_template('/Input/logistic-regression-form.html')

@app.route('/decisiontreeregressionform')
def decisiontreeregressionform():
    return render_template('/Input/decisiontree-regression-form.html')
#Input form ends






#Pickle file loading and prediction
@app.route('/randomforestregression',methods=['POST'])
def randomforestregression():
    print("hi1")
    #int_features = [int(x) for x in request.form.values()]
    int_features = request.form.getlist('name')
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = modelRandomForest.predict(final_features)
    output = round(prediction[0], 2)
    print("hi")
    return render_template('/Input/prediction.html',results=output,final_features=final_features)

@app.route('/linearregression',methods=['POST'])
def linearregression():
    print("hi1")
    #int_features = [int(x) for x in request.form.values()]
    int_features = request.form.getlist('name')
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = modelLinearRegression.predict(final_features)
    output = round(prediction[0], 2)
    print("hi")
    return render_template('/Input/prediction.html',results=output,final_features=final_features)

@app.route('/logisticregression',methods=['POST'])
def logisticregression():
    print("hi1")
    #int_features = [int(x) for x in request.form.values()]
    int_features = request.form.getlist('name')
    print(int_features)
    final_features = [np.array([int(int_features[0]),int(int_features[1]),int(int_features[2])])]
    print(final_features)
    prediction = modelLogisticRegression.predict(final_features)
    output =prediction[0]
    print("hi")
    return render_template('/Input/classification.html',results=output,final_features=final_features)


@app.route('/decisiontreeregression',methods=['POST'])
def decisiontreeregression():
    print("hi1")
    #int_features = [int(x) for x in request.form.values()]
    int_features = request.form.getlist('name')
    print(int_features)
    final_features = [np.array([int(int_features[0]),int(int_features[1]),int(int_features[2])])]
    print(final_features)
    prediction = modelDecisiontreeClassification.predict(final_features)
    output =prediction[0]
    print("hi")
    return render_template('/Input/classification.html',results=output,final_features=final_features)

if __name__ == "__main__":
    app.config['SESSION_COOKIE_SECURE'] = False
    app.run(debug=True)
    
    
 
 #C:\Users\SITS\AppData\Local\Programs\Python\Python38\Scripts