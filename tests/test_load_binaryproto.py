import tensorflow as tf
from google.protobuf import text_format

model_path = "./tf_models/mobilenet_ssd_256res_0.125_person_cat_dog.pb"
graph_def = tf.GraphDef()

with open(model_path, "rb") as f:
    graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        print (node)   