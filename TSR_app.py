from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import pyttsx3
app = Flask(__name__)

# Classes of trafic signs
classes = {0:'Speed limit (30km/h)',
            1:'Speed limit (50km/h)', 
            2:'Speed limit (60km/h)', 
            3:'Speed limit (70km/h)', 
            4:'Speed limit (80km/h)', 
            5:'Stop', 
            6:'No Entry', 
            7:'Turn Left', 
            8:'Turn Right', 
            9:'Bumpy Road'
            }

def image_processing(img):
	model = load_model('./Model/model_TSR.h5')
	data=[]
	image = Image.open(img)
	image = image.resize((30,30))
	data.append(np.array(image))
	X_test = np.array(data)
	Y_pred = model.predict(X_test)
	classes_x = np.argmax(Y_pred,axis=1)
	return classes_x

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/first', methods=['GET'])
def first():
    # Main page
    return render_template('first.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
        # Get the file from post request
		f = request.files['file']
		file_path = secure_filename(f.filename)
		f.save(file_path)
        # Make prediction
		result = image_processing(file_path)
		s = [str(i) for i in result]
		a = int("".join(s))
		result = "Predicted Traffic Sign is: " +classes[a]
		os.remove(file_path)
		engineio = pyttsx3.init()
		engineio.say(result)
		results=engineio.runAndWait()
		return result
	return results     

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")    

if __name__ == '__main__':
    app.run(port=5000,debug=True)