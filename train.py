from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils.preprocess_images as preprocess_images
import sys
import os

from distutils.errors import PreprocessError
from utils.classifier import training

PREPROCESS_FOLDERNAME = 'pre_img/'
MODEL_PATH = './model.pb'
CLASSIFIER_PATH = './classifier.pkl'

class Main(object): 
   def __init__(self):
       self.model_path = MODEL_PATH
       self.model = None
      
   def train(self, training_directory):
       preprocess_folder = self.load_data(training_directory)
       print("Training")
       obj=training(preprocess_folder, MODEL_PATH, CLASSIFIER_PATH)
       get_file=obj.main_train()
       print('Saved classifier model to file "%s"' % get_file)
       print("Training completed.")

   def evaluate(self, evaluation_directory):
       preprocess_folder = self.load_data(evaluation_directory)
       print("Evaluation")
       obj=training(preprocess_folder, MODEL_PATH, CLASSIFIER_PATH)
       eval_result=obj.main_evaluate()
       print("Evaluation completed.")
       return eval_result

   def save(self):
       #joblib.dump(self.model, self.model_path)
       print("save")

   def load_data(self, path):
       print("path: " + path)
       path = os.path.abspath(path)
       print("path: " + path)
       preprocess_folder = os.path.join(path, PREPROCESS_FOLDERNAME)
       preprocess_images.startPreprocessing(path, preprocess_folder)
       return preprocess_folder

TRAIN_FOLDER = './train_img/'
EVAL_FOLDER = './eval_img/'

# Test the ML Package locally
#m = Main()
#print(m.evaluate(EVAL_FOLDER))