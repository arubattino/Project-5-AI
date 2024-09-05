from matplotlib import units
from numpy import dtype
#from utils.data_aug import create_data_aug_layer
from tensorflow import keras
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
):

    # Create the model to be used for finetuning here!
    if weights == "imagenet":

        
        inputs = keras.layers.Input(shape=input_shape, dtype = tf.float32)
       
        preprocess_input = keras.applications.resnet50.preprocess_input
        
        base_model = keras.applications.resnet.ResNet50(weights = weights,
                                                        include_top = False,
                                                        pooling = 'avg'
                                                        )


        base_model.trainable = True
    
        
        if data_aug_layer != None:
            
            data_augmentation = keras.Sequential([layers.RandomFlip("horizontal")])
            x = data_augmentation(inputs)
            x = preprocess_input(x)
            x = base_model(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            
        else:
            x = preprocess_input(inputs)
            x = base_model(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        outputs = keras.layers.Dense(classes, kernel_regularizer='l2', activation='softmax')(x) 
        
        model = keras.Model(inputs, outputs)
    else:
        
        model = keras.models.load_model(weights)

    return model
