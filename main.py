from flask import  Flask  , jsonify, request
import pickle as pk
import numpy as np
from flask import  render_template
from flask_uploads import UploadSet , configure_uploads ,DATA
from flask import  request as r
import pandas as pd
from sklearn.preprocessing import StandardScaler , Imputer
from sklearn.pipeline import  Pipeline

num_pipeLine = Pipeline(
    [
        ('imputer', Imputer(strategy="mean")),
        ('standardScaler', StandardScaler())

    ]

)


model =  pk.load(open("model.pkl","rb"))


app = Flask(__name__)


photos = UploadSet('photos', DATA)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return filename
    return render_template('upload.html')







@app.route("/")
def home():
    return(render_template("main.html"))


@app.route("/predict")
def route():
    return  render_template("upload.html")

@app.route("/ss" ,methods=["POST"])
def predict():
    data = request.get_json(force = True)

    predict_r = [data['fixed acidity'],
                 data['volatile acidity'],
                 data['citric acid'],
                 data['residual sugar'],
                 data['chlorides'],
                 data[ 'free sulfur dioxide'],
                 data['total sulfur dioxide'],
                 data['density'],
                 data['pH'],
                 data['sulphates'],
                 data['alcohol']
                 ]
    predict_r =  np.array(predict_r)

    y_hat =  model.predict(predict_r).tolist()

    output   =  y_hat[0]

    return jsonify(results = output)



@app.route("/predict_2" , methods =["POST" ,"GET"])
def predict_2():
    data =  pd.read_csv("static/img/winequality-red.csv")
    X =  data.drop('quality' , axis=1)
    X_trans= num_pipeLine.fit_transform(X)

    data = np.array(X_trans)

    y_hat = model.predict(data).tolist()

    output = y_hat

    print("@@@@@@@@@@@@")
    return jsonify(results =  output)








if __name__ == "__main__":
    app.run(port=5000 , debug= True)




