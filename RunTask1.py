import os.path
import tensorflow
import numpy as np

from Parameters import *
from FacialDetector import *
import pdb
from Utils.Visualize import *

# TODO: change model to a CNN model (try with feature extraction & without)
# TODO: try to change where we calculate the hog image

params: Parameters = Parameters()

params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
# params.dim_hog_cell = 4  # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 5000  # numarul exemplelor pozitive
params.number_negative_examples = 50000  # numarul exemplelor negative
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

# Possible values: SVM, CNN_with_HOG, CNN
params.model_used = "CNN_with_HOG"

if not os.path.exists(params.dir_pos_examples):
    os.makedirs(params.dir_pos_examples)
if not os.path.exists(params.dir_neg_examples):
    os.makedirs(params.dir_neg_examples)
if not os.path.exists(params.dir_save_files):
    os.makedirs(params.dir_save_files)

if params.model_used == "SVM":
    # 0.62
    params.threshold = 1
    params.image_initial_scale = 1.35
    params.image_minimize_scale = 0.90
    params.step_between_windows = 1
    params.dim_hog_cell = 6  # dimensiunea celulei
    params.yellow_percentage = 0.5
    params.cells_per_block = (3, 3)
elif params.model_used == "CNN_with_HOG":
    # 0.782
    params.threshold = 1.5
    params.image_initial_scale = 1.5
    params.image_minimize_scale = 0.90
    params.step_between_windows = 1
    params.dim_hog_cell = 6  # dimensiunea celulei
    params.yellow_percentage = 0.5
    params.cells_per_block = (3, 3)
elif params.model_used == "CNN":
    params.threshold = 2
    params.image_initial_scale = 2.00
    params.image_minimize_scale = 0.90
    params.yellow_percentage = 0.3
    params.step_between_windows = 1
    params.iou_threshold = 0.20

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

detections, scores, file_names = None, None, None

print("Predicting the data:")

if params.model_used == "CNN":
    positive_images_path = params.dir_pos_examples
    negative_images_path = params.dir_neg_examples
    positive_images = facial_detector.read_images(positive_images_path)
    positive_labels = np.ones(positive_images.shape[0])
    negative_images = facial_detector.read_images(negative_images_path)
    negative_labels = np.zeros(negative_images.shape[0])

    training_examples = np.concatenate((np.squeeze(positive_images), np.squeeze(negative_images)), axis=0)
    train_labels = np.concatenate((positive_labels, negative_labels))

    facial_detector.train_classifier_CNN(training_examples, train_labels)
    detections, scores, file_names = facial_detector.run_CNN()

else:
    positive_features_path = os.path.join(params.dir_save_files,
                                              'descriptorsPositiveExamples_' + str(params.dim_hog_cell) + '_' +
                                              str(params.number_positive_examples) + '.npy')
    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        print('Am incarcat descriptorii pentru exemplele pozitive')
    else:
        print('Construim descriptorii pentru exemplele pozitive:')
        positive_features = facial_detector.get_positive_descriptors()
        np.save(positive_features_path, positive_features)
        print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

    # exemple negative
    negative_features_path = os.path.join(params.dir_save_files,
                                          'descriptorsNegativeExamples_' + str(params.dim_hog_cell) + '_' +
                                          str(params.number_negative_examples) + '.npy')
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        print('Am incarcat descriptorii pentru exemplele negative')
    else:
        print('Construim descriptorii pentru exemplele negative:')
        negative_features = facial_detector.get_negative_descriptors()
        np.save(negative_features_path, negative_features)
        print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))

    # clasificator
    if params.model_used == "SVM":
        facial_detector.train_classifier(training_examples, train_labels)
        detections, scores, file_names = facial_detector.run()
    else:
        facial_detector.train_classifier_CNN_with_HOG(training_examples, train_labels)
        detections, scores, file_names = facial_detector.run_CNN_with_HOG()

if not os.path.exists(params.evaluation_dir_task_one):
    os.makedirs(params.evaluation_dir_task_one)

np.save(os.path.join(params.evaluation_dir_task_one, "detections" + params.task_one_text), detections)
np.save(os.path.join(params.evaluation_dir_task_one, "file_names" + params.task_one_text), file_names)
np.save(os.path.join(params.evaluation_dir_task_one, "scores" + params.task_one_text), scores)

print("Files have been saved!")

# if params.has_annotations:
#     facial_detector.eval_detections(detections, scores, file_names)
#     show_detections_with_ground_truth(detections, scores, file_names, params)
# else:
#     show_detections_without_ground_truth(detections, scores, file_names, params)
