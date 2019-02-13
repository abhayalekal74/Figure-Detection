from sys import argv, exit
import argparse
import json

from os import listdir
from os.path import isfile, join

import keras
import numpy as np
from scipy.misc import imread, imresize


def die(err):
    exit("\nError: {}, exiting".format(err))


def validate_args(args):
    # It is mandatory to pass classes json. If not found, the program will exit
    if not args.classes:
        die("classes file is required")
    try:
        open(args.classes)
    except IOError:
        die("classes file not found")

    # If predicting and a model is not passed, die
    if args.predict and not args.model:
        die("Pass a model to predict")        


def parse_args():
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument("--classes", help="json containing folder path as the key and class as value")
    parser.add_argument("--model", help="pass an already trained model for further training")
    parser.add_argument("--shape", help="shape the images should be resized to")
    parser.add_argument("--save-as", dest="saveas", default="output_model.h5", help="save the model as")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--no-resize", dest="resize", action="store_false", help="all images are of same shape, no need to resize")
    parser.add_argument("--pred", dest="predict", action="store_true", help="run prediction instead of training")
    parser.set_defaults(predict=False)
    parser.set_defaults(resize=True)
    args = parser.parse_args()
    validate_args(args)
    return args


def get_images(classes):
    images = []
    for folder_name in classes:
        images += [join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]
    return images


def get_classes(classes_json_file):
    with open(classes_json_file) as classes_json:
        classes = json.load(classes_json)
    return classes


def get_resize_shape(images, shape=None):
    # If the shape is provided, use as it is else calculate mean rows and cols. 
    if shape:
        rows, cols = map(int, shape.split(","))
    else:    
        row_shapes, col_shapes = [], []
        for img in images:
            img_matrix = imread(img, mode='RGB')
            if img_matrix is not None:
                row_shapes.append(img_matrix.shape[0])
                col_shapes.append(img_matrix.shape[1])
        rows = (int) (np.mean(row_shapes))
        cols = (int) (np.mean(col_shapes))
        print ("Resizing images to shape [{},{}]".format(rows, cols))
    return rows, cols


class FigureDetector:

    def __init__(self, args):

        self.classes = get_classes(args.classes)
        self.images = get_images(self.classes)

        self.model_load_from = args.model
        self.save_as = args.saveas
        self.epochs = args.epochs
        self.should_resize_images = args.resize

        if self.should_resize_images:
            self.rows, self.cols = get_resize_shape(self.images, args.shape)

        self.encode_labels()
        self.build_model()


    def encode_labels(self):
        from sklearn.preprocessing import MultiLabelBinarizer 
        labels = []
        # labels in string form are encoded using MultiLabelBinarizer
        for k, v in self.classes.items():
            labels.append([l.strip() for l in v.split(",")])
        self.label_encoder = MultiLabelBinarizer()
        print ("Transforms: {}".format(self.label_encoder.fit_transform(labels)))        
        print ("Output classes: {}".format(self.label_encoder.classes_))
 

    def get_dataset(self, start, end):
        x,y = [],[]
        for i in range(start, end):
            img = self.images[i]
            # Read the image as a matrix in RGB mode
            img_matrix = imread(img, mode='RGB')
            if img_matrix is not None:
                if self.should_resize_images:
                    # Resize the images to a common shape
                    img_matrix = imresize(img_matrix, (self.rows, self.cols, 3)) 
                # Append the image matrix to the list of input matrices
                x.append(img_matrix)
                labels = str(self.classes[img[:img.rfind("/")]]).split(",")
                # Append the labels to output list
                y.append(np.squeeze(self.label_encoder.transform([labels])))
       
            # Normalize
        x -= np.mean(x)
        x /= np.std(x)
 
        return np.array(x), np.array(y)


    def build_model(self):
        if self.model_load_from:
            # If a model has been passed, use it.
            self.model = keras.models.build_model(self.model_load_from)
        else:
            # Otherwise build a CNN
            self.model = keras.models.Sequential()
    
            # Third set of layers
            self.model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
            self.model.add(keras.layers.MaxPooling2D(pool_size=(3,3), padding='same'))
            self.model.add(keras.layers.Dropout(0.25))
            
            # Third set of layers
            self.model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
            self.model.add(keras.layers.MaxPooling2D(pool_size=(3,3), padding='same'))
            self.model.add(keras.layers.Dropout(0.25))
            
            # Third set of layers
            self.model.add(keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
            self.model.add(keras.layers.MaxPooling2D(pool_size=(3,3), padding='same'))
            self.model.add(keras.layers.Dropout(0.25))
            
            # Third set of layers
            self.model.add(keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
            self.model.add(keras.layers.MaxPooling2D(pool_size=(3,3), padding='same'))
            self.model.add(keras.layers.Dropout(0.25))
            
            # Fourth set of layers
            self.model.add(keras.layers.Conv2D(128, (3, 3), activation='relu')) 
            self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
            self.model.add(keras.layers.Dropout(0.25))

            # Flattening the input to be passed onto Fully Connected Layers
            self.model.add(keras.layers.Flatten())

            # First fully connected layer
            self.model.add(keras.layers.Dense(128, activation='relu'))
            self.model.add(keras.layers.Dropout(0.5))

            # Last layer, responsible for predicting the output
            self.model.add(keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax'))

    
    def train(self):
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Create checkpoint after every epoch
        cb = [keras.callbacks.ModelCheckpoint(self.save_as[:-3] + "_cp.h5", monitor='acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)]
        
        x_train, y_train = self.get_dataset(5, len(self.images))
        x_val, y_val = self.get_dataset(0, 5)
        
        # Fit the model on the training data
        self.model.fit(x_train, y_train,
                  epochs=self.epochs,
                  batch_size=32,
                  callbacks=cb
                )   

        # Save the model with the value passed for -saveas argument
        self.model.save(self.save_as)

        print ("\nEvaluation on validation data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(x_val, y_val, batch_size=32)))))


    def predict(self):
        x_test, y_test = self.get_dataset(0, len(self.images))
        res = self.model.predict(x_test, batch_size=32, verbose=1)    
        class_argmax = {} # Storing encoding index
        for c in self.label_encoder.classes_:
            class_argmax[np.argmax(self.label_encoder.transform([[c,]]))] = c
        for i in range(len(self.images)):
            print (self.images[i], class_argmax[np.argmax(res[i])])
        print ("\nEvaluation on test data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(x_test, y_test, batch_size=32)))))


if __name__=="__main__":
    args = parse_args()

    figure_detector = FigureDetector(args)
    figure_detector.predict() if args.predict else figure_detector.train()
