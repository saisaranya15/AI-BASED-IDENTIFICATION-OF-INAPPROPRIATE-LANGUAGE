import tensorflow as tf
import numpy as np

with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()


loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loaded_model.load_weights("model_weights.weights.h5")


import joblib

RF=joblib.load("RF.sav")
DE=joblib.load("DT.sav")

from sklearn.feature_extraction.text import HashingVectorizer

hvectorizer = HashingVectorizer(n_features=10000, norm=None, alternate_sign=False, stop_words='english')


from flask import Flask, render_template, request, url_for,send_from_directory,Response


list1=['Hate Speech','Not offensive language','offensive language']



app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'

@app.route('/')
def home():
    return render_template('file.html',prediction=None)



@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        text=request.form["text"]
        model=request.form["name"]
        if text.strip() == "" :
            return render_template('file.html', prediction=None,msg="text field required")
        else:
            text=text.strip()
            values=hvectorizer.fit_transform([text]).toarray()
            if model == "RandomForest":
                prediction=RF.predict(values)
            elif model == "DecisionTree":
                prediction=DE.predict(values)
            elif model == "Lstm":
                prediction=loaded_model.predict(values)
                prediction=np.argmax(prediction)
                return render_template('file.html', prediction=list1[prediction])

        return render_template('file.html', prediction=list1[prediction[0]])

    return render_template('file.html', prediction=None)








if __name__ == '__main__':
    app.run(debug=True)
