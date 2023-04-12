from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import cv2
import numpy as np
import os
import time
import pickle
import sys
import utils.facenet as facenet
import utils.detect_face as detect_face
import tensorflow.compat.v1 as tf
import io
from PIL import Image

tf.disable_v2_behavior()

MODEL_PATH = './model.pb'
CLASSIFIER_PATH = './classifier.pkl'
NPY_PATH = './npy'

class Main(object):
   def __init__(self):
      self.graph = tf.Graph().as_default()

   def predict(self, input):
       
       inputImage = Image.open(io.BytesIO(input))
       result_names = ''

       with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, NPY_PATH)

                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                frame_interval = 3
                batch_size = 1000
                image_size = 182
                input_image_size = 160
           

                print('Loading feature extraction model...')
                facenet.load_model(MODEL_PATH)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]


                classifier_filename_exp = os.path.expanduser(CLASSIFIER_PATH)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, classes) = pickle.load(infile, encoding='latin1')

                HumanNames = classes[:-1]
                # video_capture = cv2.VideoCapture("akshay_mov.mp4")
                c = 0


                print('Start Recognition...')
                prevTime = 0

                #frame = cv2.imread(img_name,0)
                nimg = np.array(inputImage)
                frame = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                curTime = time.time()+1    # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Faces Detected: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is too close')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            #scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
   
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            
                            predictions = model.predict_proba(emb_array)
                            print(predictions)

                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            print(best_class_probabilities)

                            if len([x for x in predictions[0].tolist() if x >= 0.8]) == 0:
                                return 'Error: No valid faces detected.'

                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #Get Human name of detected face
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            print('Result Indices: ', best_class_indices[0])
                            print(HumanNames)
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names += HumanNames[best_class_indices[0]] + ";"
                            result_names = result_names[:-1]
                            print("Valid faces detected: " + result_names)

                    else:
                        return 'Error: No faces detected.'

       return result_names



# Test the ML Package locally
#with open('./capture.png', 'rb') as input_file:
#    bytes = input_file.read()
#m = Main()
#print(m.predict(bytes))