import os

import numpy as np


class Parameters:
    def __init__(self):
        self.base_dir = 'data/'
        self.test_dir = 'validare/'
        self.dir_pos_examples = os.path.join(self.base_dir, 'positiveExamples')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negativeExamples')
        self.dir_test_examples = os.path.join(self.test_dir, 'simpsons_validare/')
        self.path_annotations = os.path.join(self.test_dir, 'simpsons_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'savedFiles')
        self.model_used="CNN"
        self.image_initial_scale = 1.50
        self.image_minimize_scale = 0.90
        self.step_between_windows = 1
        self.iou_threshold = 0.20
        self.hsv_color1 = np.asarray([20, 30, 30])
        self.hsv_color2 = np.asarray([85, 255, 255])
        # self.window_proportions = [(36, 36), (32, 40), (28, 45)]
        # self.window_proportions = [(36, 36), (36, 45), (36, 56)]
        self.window_scales = [(1, 1), (1, 0.9), (1, 0.8), (1, 0.7), (1, 0.65)]
        # self.image_initial_scale = 1.35
        # self.image_minimize_scale = 0.90
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.yellow_percentage = 0.5
        self.cells_per_block = (3, 3)
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 5453  # numarul exemplelor pozitive
        self.number_negative_examples = 44039  # numarul exemplelor negative
        self.threshold = 0  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        self.has_annotations = True

        self.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
        self.use_flip_images = False  # adauga imaginile cu fete oglindite
