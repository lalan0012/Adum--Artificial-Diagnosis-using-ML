from flask import Flask
#from flask_ngrok import run_with_ngrok
app = Flask(__name__,static_folder="/myCSS")
#run_with_ngrok(app)
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from flask import redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
import sys
import shutil
from flask_cors import CORS, cross_origin
import tensorflow as tf 
from uuid import uuid4
cors = CORS(app, resources={r"/*": {"origins": "*"}}) 
y=[]
#@app.route('/', methods=['GET'])
#def index():
# Main page
#return render_template('index.html')
#
@app.route('/covidPage.html', methods=['GET', 'POST'])
#@cross_origin()
def predict():
  prediction=""
  if request.method=="POST":
    f = request.files['file']
    # Save the file to ./uploads
    #basepath = os.path.dirname(__file__)
    image1 = secure_filename(f.filename) #os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(image1)
    #rem('.\uploads')
    print("done")
    print('model loading ...')
    covid_model = load_model('Covid_model.h5',compile=False)
    print('model loading done.')
    #xray_model = load_model("/content/xrayornot_data/xrayornot_model2.h5")
    test_image = image.load_img(image1,target_size=(224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    results = covid_model.predict(test_image)
    #result = xray_model.predict(test_image)
    #if it is an xray then
    if np.argmax(results, axis=1) == 0 :# and result2[0][0]>4.226988e-15:
        prediction = 'High risk of COVID-19'
    else:
      prediction = 'Patient is Healthy'
    print('===================================')
    print(prediction)
    print('===================================')
    print("inside if")
    return render_template('/covidPage.html',resultt=prediction)
  else:
    print("inside else")
    return render_template('/covidPage.html',resultt=prediction)
@app.route('/views/brainTumourPage.html', methods=['GET', 'POST'])
#@cross_origin()
def predict2():
  prediction=""
  if request.method=="POST":
    f = request.files['file']
    # Save the file to ./uploads
    #basepath = os.path.dirname(__file__)
    image1 = secure_filename(f.filename) #os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(image1)
    #rem('.\uploads')
    print("done")
    print('model loading ...')
    covid_model = load_model('Brain_model.h5',compile=False)
    print('model loading done.')
    #xray_model = load_model("/content/xrayornot_data/xrayornot_model2.h5")
    test_image = image.load_img(image1,target_size=(224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    results = covid_model.predict(test_image)
    #result = xray_model.predict(test_image)
    #if it is an xray then
    if np.argmax(results, axis=1) == 0 :# and result2[0][0]>4.226988e-15:
        prediction = 'High risk of brainTumour'
    else:
      prediction = 'Patient is Healthy'
    print('===================================')
    print(prediction)
    print('===================================')
    print("inside if")
    return render_template('/views/brainTumourPage.html',resultt=prediction)
  else:
    print("inside else")
    return render_template('/views/brainTumourPage.html',resultt=prediction)



if __name__=="__main__":
  #http_server = WSGIServer(('',8080),app)
  #http_server.serve_forever()
  app.run()
