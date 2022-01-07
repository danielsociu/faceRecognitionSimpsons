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
params.threshold = 2  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)
# exemple pozitive
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

detections, scores, file_names = None, None, None
# clasificator
if params.model_used == "SVM":
    facial_detector.train_classifier(training_examples, train_labels)
    detections, scores, file_names = facial_detector.run()
else:
    facial_detector.train_classifier_CNN_with_HOG(training_examples, train_labels)
    detections, scores, file_names = facial_detector.run_cnn()

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
