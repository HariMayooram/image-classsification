import tensorflow as tf
import tensorflow_hub as hub
import warnings
import argparse 
warnings.filterwarnings('ignore')
import logging 

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image 
import json
import numpy as np 


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', default='./test_images/cautleya_spicata.jpg', help = 'Image path in order to get the prediction', type = str)
parser.add_argument('--model' ,default='./best_model.h5' , help = 'Keras model path which has been saved', type = str)
parser.add_argument('--top_k', default=5, help = 'The number of predictions requried', type = int)
parser.add_argument ('--category_names' , default = 'label_map.json', help = 'Categories for file path which is in JSON format', type = str)
args = parser.parse_args()


image_path =args.image_path
model = args.model
top_k =args.top_k
class_label_map =args.category_names


with open(class_label_map, 'r') as f:
    class_names = json.load(f)
        
        
        
LoadingModel = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})

def imageProcessing(test_obj):
    processed_part_of_image = np.squeeze(test_obj)
    processed_part_of_image = tf.image.resize(processed_part_of_image, (224,224))
    processed_part_of_image /=255
    return processed_part_of_image


def predicting_imgsec(image_path, model, top_k, class_label_map):
    image_detls = Image.open(image_path)
    transform_image = imageProcessing(np.asanyarray(image_detls))
    prediction = model.predicting_imgsec(np.expand_dims(transform_image,axis=0))
    prop, indices = tf.math.top_k(prediction, k=top_k)
    prop = prop.numpy()[0]
    name_of_flower = [class_names[str(value+1)] for value in indices.cpu().numpy()[0]]
    return prop, name_of_flower


if __name__ == '__main__':
    print(predicting_imgsec(image_path,LoadingModel , top_k, class_names))
