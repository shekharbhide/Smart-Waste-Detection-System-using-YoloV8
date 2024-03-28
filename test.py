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

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)
                                               
            file_extension = f.filename.rsplit('.', 1)[1].lower() 
            
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                

                image = Image.open(io.BytesIO(frame))

                # Perform the detection
                save_dir='runs/detect'
                yolo = YOLO('best.pt')
                detections = yolo.predict(image, save=True ,project=r'runs\detect')
                print('save_dir : ' + save_dir)
                return display(f.filename)
            
            elif file_extension == 'mp4': 
                video_path = filepath  # replace with your video path
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
                    
                    
                    
                    '''
                    For each frame of the video, the YOLOv8 model performs object detection using the model object and returns the results in a list.
                    Each element in the list contains the class ID, confidence score, and bounding box coordinates for a detected object.
                    '''
                    # do YOLOv8 detection on the frame here
                    #model = YOLO('best.pt')
                    results = model(frame, save=True)  #working
                    print(results)
                    cv2.waitKey(1)
                    
                    
                    '''
                    You can use plot() function of Result object to plot results on in image object. 
                    It plots all components(boxes, masks, classification logits, etc.) found in the results object
                    '''
                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)
                    
                    # write the frame to the output video
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break
                    
                    # for result in results:
                        # #class_id, confidence, bbox = result
                        # boxes = result.boxes  # Boxes object for bbox outputs
                        # probs = result.probs  # Class probabilities for classification outputs
                        # cls = boxes.cls 
                        # xyxy = boxes.xyxy 
                        # xywh = boxes.xywh  # box with xywh format, (N, 4)
                        # conf = boxes.conf
                        # print("boxes : ",boxes)
                        # print("probs : ",probs)     
                        # print("cls - cls, (N, 1) : ",cls)      
                        # print("conf - confidence score, (N, 1): ",conf)
                        # print("box with xyxy format, (N, 4) : ",xyxy)                       
                        # print("box with xywh format, (N, 4) : ",xywh) 


                return video_feed()            


            
    folder_path = r'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    print(image_path)
    return render_template('index.html', image_path=image_path)
    #return "done"



# #The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = r'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder    
    print("printing directory: ",directory) 
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

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

        '''The yield keyword works similar to return, but instead of returning a single value, 
        it returns a sequence of values that can be iterated over.
         In this function, yield keyword is used to return a stream of JPEG-encoded image frames. 
        '''        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")
    '''
    returns a Flask Response object that streams the video frames
    The get_frame() function is a generator that continuously reads frames from the input video, performs object detection using the YOLOv8 model,
    and returns the annotated frames as JPEG images in a multi-part HTTP response. 
    '''
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        


# Function to start webcam and detect objects

'''
the cv2.VideoCapture(0) line creates a VideoCapture object that is connected to the default camera (usually the first webcam). 
The generate() function captures frames from the camera using video.read(), and yields each frame to the client as a multi-part HTTP response.
 The Response object returned by the route function specifies the generator and the mimetype of the response.

You can access the video feed in your web browser by visiting http://localhost:5000/webcam_feed.
'''

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
                    
                    
            '''
            You can use plot() function of Result object to plot results on in image object. 
            It plots all components(boxes, masks, classification logits, etc.) found in the results object
            '''
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


       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    #model = torch.hub.load('.', 'custom','best.pt', source='local')
    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port) 