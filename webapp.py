
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob


from ultralytics import YOLO


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')

from flask import jsonify

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if 'file' in request.files:
        f = request.files['file']
        if f.filename == '':
            return redirect(request.url)

        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension == 'jpg':
            img = cv2.imread(filepath)

            # Specify the save directory
            save_path = r'runs\detect'

            # Perform the detection
            yolo = YOLO('best.pt')
            detections = yolo.predict(img, save=True, project=r'runs\detect')
            
            return display(f.filename)
        
        elif file_extension == 'mp4': 
            video_path = filepath  
            cap = cv2.VideoCapture(video_path)

            # get video dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
            
            # initialize the YOLOv8 model here
            model = YOLO('best.pt')
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break                                                      

                # do YOLOv8 detection on the frame here
                model = YOLO('best.pt')
                results = model(frame, save=True,project=r'runs\detect')  #working
                print(results)
                cv2.waitKey(1)

                res_plotted = results[0].plot()
                cv2.imshow("result", res_plotted)
                
                # write the frame to the output video
                out.write(res_plotted)

                if cv2.waitKey(1) == ord('q'):
                    break

            return jsonify({'success': True})  # Return JSON response indicating success

    # If no file is uploaded or processed, return JSON response with error
    return jsonify({'success': False, 'error': 'No file processed'})


@app.route('/get_latest_image_path')
def get_latest_image_path():
    folder_path = r'runs\detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.startswith('predict')]
    if subfolders:
        numeric_subfolders = [int(subfolder.split('predict')[1]) for subfolder in subfolders if subfolder.split('predict')[1].isdigit()]
        if numeric_subfolders:
            latest_subfolder = 'predict' + str(max(numeric_subfolders))
            folder_path = os.path.join(folder_path, latest_subfolder)
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                image_path = os.path.join(folder_path, latest_file)
                return jsonify({'success': True, 'image_path': image_path})
            else:
                return jsonify({'success': False, 'error': 'No files found in the latest subfolder'})
        else:
            return jsonify({'success': False, 'error': 'No numeric subfolders found in the specified directory'})
    else:
        return jsonify({'success': False, 'error': 'No subfolders found in the specified directory'})


@app.route('/display_image')
def display_image():
    return render_template('index.html')


# The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = r'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder    
    print("printing directory: ", directory) 
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ) #shows the result in separate tab

    else:
        return "Invalid file format"
     
        
 
       

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        






# function for accessing rtsp stream
# modify the rtsp stream as per your camera
@app.route("/rtsp_feed")
def rtsp_feed():
    cap = cv2.VideoCapture('rtsp://username:password@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()
            print(type(frame))
            
            img = Image.open(io.BytesIO(frame))
            #results = model(img, size=640)      
   
            model = YOLO('best.pt')
            results = model(img, save=True)              

            print(results)
            cv2.waitKey(1)

            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)
                    
            # write the frame to the output video
            #out.write(res_plotted)

            if cv2.waitKey(1) == ord('q'):
                break

            # read image as BGR
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR) 
            
            # Encode BGR image to bytes so that cv2 will convert to RGB
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            #print(frame)
                

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')






# Function to start webcam and detect objects

@app.route("/webcam_feed")
def webcam_feed():
    #source = 0
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()
            print(type(frame))
            
            img = Image.open(io.BytesIO(frame))
 
            
            model = YOLO('best.pt')
            results = model(img, save=True)              

            print(results)
            cv2.waitKey(1)

            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)


            if cv2.waitKey(1) == ord('q'):
                break

            # read image as BGR
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR) 
            
            # Encode BGR image to bytes so that cv2 will convert to RGB
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            #print(frame)
                

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    #model = torch.hub.load('.', 'custom','best.pt', source='local')
    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port) 
