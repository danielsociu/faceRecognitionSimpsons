import numpy as np
from collections import defaultdict

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
params.model_used = "CNN"

params.threshold = 2
params.image_initial_scale = 2.00
params.image_minimize_scale = 0.90
params.yellow_percentage = 0.3
params.step_between_windows = 1
params.iou_threshold = 0.20

def encode_labels(my_labels):
    number_labels = []
    for i in range(len(my_labels)):
        number_labels.append(int(params.encoding[my_labels[i]]))
    number_labels = np.array(number_labels)
    return number_labels

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

positive_images_path = params.dir_pos_examples
train_images, train_labels = facial_detector.read_characters()

number_train_labels = encode_labels(train_labels)


facial_detector.train_recognition_classifier(train_images, number_train_labels)
detections, scores, file_names, names = facial_detector.run_recognition()

data = defaultdict(lambda: defaultdict(lambda: []))

for index in range(names.shape[0]):
    reversed_encoding = {value: key for (key, value) in params.encoding.items()}
    name = reversed_encoding[names[index]]
    data[name]['detections'].append(detections[index])
    data[name]['file_names'].append(file_names[index])
    data[name]['scores'].append(scores[index])

names = ['bart', 'homer', 'lisa', 'marge']
if not os.path.exists(params.evaluation_dir_task_two):
    os.makedirs(params.evaluation_dir_task_two)
for name in names:
    np.save(os.path.join(params.evaluation_dir_task_two, "detections_" + name + ".npy"), np.array(data[name]['detections']))
    np.save(os.path.join(params.evaluation_dir_task_two, "file_names_" + name + ".npy"), np.array(data[name]['file_names']))
    np.save(os.path.join(params.evaluation_dir_task_two, "scores_" + name + ".npy"), np.array(data[name]['scores']))



# facial_detector.train_classifier_CNN(train_images, train_labels)
# detections, scores, file_names = facial_detector.run_CNN()
#
# # np.save(detections)
# if params.has_annotations:
#     facial_detector.eval_detections(detections, scores, file_names)
#     show_detections_with_ground_truth(detections, scores, file_names, params)
# else:
#     show_detections_without_ground_truth(detections, scores, file_names, params)
