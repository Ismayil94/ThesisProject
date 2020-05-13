from picamera.array import PiRGBArray
from picamera import PiCamera
from multiprocessing import Process
from multiprocessing import Queue
import time
import cv2
import numpy as np
import tensorflow as tf
import sys

DEBUG = True

camera = PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1920, 1080))

time.sleep(0.1)

labelss = [line.rstrip() for line 
                   in tf.gfile.GFile("./retrained_data/retrained_labels.txt")]
with tf.gfile.FastGFile("./retrained_data/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

faceDetect = cv2.CascadeClassifier('./retrained_data/haarcascade_frontalface_default.xml')


def prediction_fn(model, inputQueue, outputQueue, labelss):
    while True:
        if not inputQueue.empty():
            start1 = time.time()
            input_image = inputQueue.get()
			
			
            predictions = sess.run(model, {'DecodeJpeg:0': input_image})
            prediction = predictions[0]

            prediction = prediction.tolist()
            max_value = max(prediction)
            max_index = prediction.index(max_value)
            predicted_label = labelss[max_index]

            end1 = time.time()
            print("%s (%.2f%%) %.4f" % (predicted_label, max_value * 100, end1-start1))
            outputQueue.put(predicted_label)

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
predictions = None

p = Process(target=prediction_fn, args=(softmax_tensor, inputQueue,
	outputQueue,labelss,))
p.daemon = True
p.start()

i=0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    i = i + 1
    start = time.time()
    
    image = frame.array
    cv2.imshow("face", image)
    if DEBUG:
        print (image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        print (gray.shape)
    faces = faceDetect.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5
                                        )
    end = time.time()
    if (i<10):
        print ("Frame detection time: " + (end - start))

    for (x,y,w,h) in faces:
        if inputQueue.empty():
            inputQueue.put(gray[y:y+h,x:x+w])

        if not outputQueue.empty():
            predictions = outputQueue.get()

        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)

    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break



