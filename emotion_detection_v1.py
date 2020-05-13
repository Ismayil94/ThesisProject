from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import tensorflow as tf
import sys

debug = True

camera = PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(192time.sleep(0.1)

labelss= [line.rstrip() for line 
                   in tf.gfile.GFile("./retrained_data/retrained_labels.txt")]
with tf.gfile.FastGFile("./retrained_data/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

faceDetect = cv2.CascadeClassifier('./retrained_data/haarcascade_frontalface_default.xml')

sess = tf.Session()
softmaxten = sess.graph.get_tensor_by_name('final_result:0')

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    image = frame.array
    cv2.imshow("face", image)
    if debug:
        print (image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        print (gray.shape)
    faces = faceDetect.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5
                                        )


    for (x,y,w,h) in faces:
      
        predictions = sess.run(softmaxten, {'DecodeJpeg:0': gray[y:y+h,x:x+w]})
        prediction = predictions[0]

        prediction = prediction.tolist()
        max_value = max(prediction)
        max_index = prediction.index(max_value)
        predicted_label = label_lines[max_index]

        print("%s (%.2f%%)" % (predicted_label, max_value * 100))

        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)

    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break

