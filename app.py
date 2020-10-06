from __future__ import division, print_function
# coding=utf-8
import sys
import os
import PIL
import glob
import re
import numpy as np
import gdown


# Keras
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'

# Model saved with Keras model.save()
url = 'https://drive.google.com/uc?id=1Ql7jwGUl1EeYOoPAxs7u3FUNrHgU9Zlz'
output = 'ful_skin_cancer_model.h5'
gdown.download(url, output, quiet=False)
MODEL_PATH = 'Virtual_doc/ful_skin_cancer_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
     x=image.load_img(img_path)
     x=x.resize((224,224))
     x = np.expand_dims(x, axis=0)
     x=np.asarray(x)
     preds = model.predict(x)
     return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
        return render_template('skin.html')
    


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['skinpic']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        filename = secure_filename(file.filename)
        file_path = os.path.join(
            basepath, 'static', 'uploads', filename)
        file.save(file_path)
        

        # Make prediction
        preds = model_predict(file_path, model)
        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        nv = round(round(preds[0,0], 5)*100, 2)
        mel = round(round(preds[0,1], 5)*100, 2)
        bkl = round(round(preds[0,2], 2)*100, 2)
        bcc = round(round(preds[0,3], 2)*100, 2)
        akiec = round(round(preds[0,4], 2)*100, 2)
        vasc = round(round(preds[0,5], 2)*100, 2)
        df = round(round(preds[0,6], 2)*100, 2)
        result = np.array([nv, mel, bkl, bcc, akiec, vasc, df])


        # Process your result for human
     #  pred_class = preds.argmax(axis=-1)            # Simple argmax
    #   pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
     #  result = str(pred_class[0][0][1])               # Convert to string
        #return result
        return render_template('predict.html', data=result)
    
    

@app.route('/find', methods=['POST'])
def find():
         query=request.form.get('query')
         query= "skin clincs in" + query
         s = []

         try: 
            from googlesearch import search 
         except ImportError:  
            print("No module named 'google' found") 
        
         # to search 
         for j in search(query, tld="co.in", num=5, stop=6, pause=2): 
             s+=[j]
         a=s[0]
         b=s[1]
         c=s[2]
         d=s[3]
         e=s[4]
         g=s[5]
         c=np.array([a, b, c, d, e, g])
         return render_template('blank.html', data=c)
    
 
@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename) 

if __name__ == '__main__':
    app.run(debug=True)
