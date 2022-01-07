import cv2
import pandas as pd
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers

from Parameters import *
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None
        self.neural_model = None

    def get_positive_descriptors(self):
        print("AM GENERAT MAI INTAI EXEMPLELE POZITIVE")

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=self.params.cells_per_block, feature_vector=True)

            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=self.params.cells_per_block, feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        print("AM GENERAT MAI INTAI EXEMPLELE NEGATIVE")

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=self.params.cells_per_block, feature_vector=True)

            negative_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=self.params.cells_per_block, feature_vector=True)
                negative_descriptors.append(features)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels, ignore_restore=True):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name) and ignore_restore:
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c, penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                              multi_class='ovr', fit_intercept=True, intercept_scaling=2, verbose=True)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def train_classifier_CNN(self, training_examples, train_labels, ignore_restore=True):
        cnn_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d_CNN' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(cnn_file_name) and ignore_restore:
            self.neural_model = keras.models.load_model(cnn_file_name)
            return

        data = np.hstack((training_examples, train_labels.reshape(len(train_labels), 1)))
        np.random.shuffle(data)
        percentage = 80
        partition = int(len(training_examples) * percentage / 100)
        x_train, x_test = data[:partition, :-1], data[partition:, :-1]
        y_train, y_test = data[:partition, -1:].ravel(), data[partition:, -1:].ravel()

        neural_model = keras.Sequential()
        # neural_model.add(keras.Input(batch_input_shape=(64, 1296)))
        neural_model.add(layers.Dense(128, input_shape=(len(training_examples[0]), ), activation='relu'))
        neural_model.add(layers.Dense(64, activation='relu'))
        neural_model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(0.01)))
        neural_model.add(layers.Activation('linear'))
        neural_model.summary()

        neural_model.compile(loss='hinge',
                             optimizer='adam',
                             metrics=['accuracy'])

        result = neural_model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=5,
            # shuffle=True,
            verbose=True,
            use_multiprocessing=True,
        )

        neural_model.save(cnn_file_name)
        self.neural_model = neural_model

        print("Accuracy: " + str(np.array(result.history['accuracy'])))
        print("Val accuracy:" + str(np.array(result.history['val_accuracy'])))

    def train_classifier_CNN_with_HOG(self, training_examples, train_labels, ignore_restore=True):
        cnn_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d_CNN_with_HOG' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(cnn_file_name) and ignore_restore:
            self.neural_model_with_HOG = keras.models.load_model(cnn_file_name)
            return

        data = np.hstack((training_examples, train_labels.reshape(len(train_labels), 1)))
        np.random.shuffle(data)
        percentage = 80
        partition = int(len(training_examples) * percentage / 100)
        x_train, x_test = data[:partition, :-1], data[partition:, :-1]
        y_train, y_test = data[:partition, -1:].ravel(), data[partition:, -1:].ravel()

        neural_model = keras.Sequential()
        # neural_model.add(keras.Input(batch_input_shape=(64, 1296)))
        neural_model.add(layers.Dense(128, input_shape=(len(training_examples[0]), ), activation='relu'))
        neural_model.add(layers.Dense(64, activation='relu'))
        neural_model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(0.01)))
        neural_model.add(layers.Activation('linear'))
        neural_model.summary()

        neural_model.compile(loss='hinge',
                             optimizer='adam',
                             metrics=['accuracy'])

        result = neural_model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=5,
            # shuffle=True,
            verbose=True,
            use_multiprocessing=True,
        )

        neural_model.save(cnn_file_name)
        self.neural_model_with_HOG = neural_model

        print("Accuracy: " + str(np.array(result.history['accuracy'])))
        print("Val accuracy:" + str(np.array(result.history['val_accuracy'])))
        # print('Performanta clasificatorului optim pt c = %f' % best_c)
        # # salveaza clasificatorul
        # pickle.dump(best_model, open(cnn_file_name, 'wb'))

        # # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        # scores = best_model.decision_function(training_examples)
        # self.best_model = best_model
        # positive_scores = scores[train_labels > 0]
        # negative_scores = scores[train_labels <= 0]
        #
        # plt.plot(np.sort(positive_scores))
        # plt.plot(np.zeros(len(negative_scores) + 20))
        # plt.plot(np.sort(negative_scores))
        # plt.xlabel('Nr example antrenare')
        # plt.ylabel('Scor clasificator')
        # plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        # plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        # plt.show()

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        # print(image_detections[:, 2], image_detections[:, 3])
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        # iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > self.params.iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array(
            [])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            original_image = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            img = cv.resize(original_image, (0, 0), fx=self.params.image_initial_scale,
                            fy=self.params.image_initial_scale, interpolation=cv.INTER_CUBIC)
            color_img = cv.imread(test_files[i], cv.IMREAD_COLOR)
            color_img = cv.resize(color_img, (0, 0), fx=self.params.image_initial_scale,
                                  fy=self.params.image_initial_scale, interpolation=cv.INTER_CUBIC)
            power = 0

            image_scores = []
            image_detections = []

            while min(img.shape[0], img.shape[1]) > self.params.dim_window:
                for fx, fy in self.params.window_scales:
                    distorted_img = cv.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv.INTER_AREA)
                    if min(distorted_img.shape[0], distorted_img.shape[1]) < self.params.dim_window:
                        break
                    distorted_color_img = cv.resize(color_img, (0, 0), fx=fx, fy=fy, interpolation=cv.INTER_AREA)
                    hog_descriptor = hog(distorted_img,
                                         pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                         cells_per_block=self.params.cells_per_block, feature_vector=False)
                    whole_masked_image = cv.inRange(cv.cvtColor(distorted_color_img, cv.COLOR_BGR2HSV),
                                                    self.params.hsv_color1, self.params.hsv_color2)
                    num_cols = distorted_img.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = distorted_img.shape[0] // self.params.dim_hog_cell - 1
                    num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - (
                                self.params.cells_per_block[0] - 1)

                    for y in range(0, num_rows - num_cell_in_template, self.params.step_between_windows):
                        for x in range(0, num_cols - num_cell_in_template, self.params.step_between_windows):
                            x_index_image = x * self.params.dim_hog_cell
                            y_index_image = y * self.params.dim_hog_cell
                            descr = hog_descriptor[y: y + num_cell_in_template, x: x + num_cell_in_template].flatten()
                            # masked_img = cv.inRange(cv.cvtColor(distorted_color_img[y_index_image: y_index_image + self.params.dim_window, x_index_image:x_index_image + self.params.dim_window], cv.COLOR_BGR2HSV),
                            #                         self.params.hsv_color1, self.params.hsv_color2)
                            window_masked_img = whole_masked_image[
                                                y_index_image: y_index_image + self.params.dim_window,
                                                x_index_image:x_index_image + self.params.dim_window]
                            # print(distorted_hsv_img[y: y + self.params.dim_window, x:x + self.params.dim_window].shape)
                            counter_yellow = (window_masked_img == 255).sum()
                            score = np.dot(descr, w)[0] + bias
                            # score = self.neural_model.predict(np.array([descr, ]))[0]
                            # if score > self.params.threshold:
                            if score > self.params.threshold and counter_yellow > (
                                    self.params.dim_window ** 2) * self.params.yellow_percentage:
                                actual_zoom = 1 / (
                                        self.params.image_initial_scale * (self.params.image_minimize_scale ** power))
                                x_min = int((x * self.params.dim_hog_cell / fx) * actual_zoom)
                                y_min = int((y * self.params.dim_hog_cell / fy) * actual_zoom)
                                x_max = int(
                                    ((x * self.params.dim_hog_cell + self.params.dim_window) / fx) * actual_zoom)
                                y_max = int(
                                    ((y * self.params.dim_hog_cell + self.params.dim_window) / fy) * actual_zoom)
                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)
                img = cv.resize(img, (0, 0), fx=self.params.image_minimize_scale, fy=self.params.image_minimize_scale,
                                interpolation=cv.INTER_AREA)
                color_img = cv.resize(color_img, (0, 0), fx=self.params.image_minimize_scale,
                                      fy=self.params.image_minimize_scale,
                                      interpolation=cv.INTER_AREA)
                power += 1

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores),
                                                                              original_image.shape)
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_file_name = ntpath.basename(test_files[i])
                image_names = [short_file_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names

    def run_cnn(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array([])
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            original_image = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            img = cv.resize(original_image, (0, 0), fx=self.params.image_initial_scale,
                            fy=self.params.image_initial_scale, interpolation=cv.INTER_CUBIC)
            color_img = cv.imread(test_files[i], cv.IMREAD_COLOR)
            color_img = cv.resize(color_img, (0, 0), fx=self.params.image_initial_scale,
                                  fy=self.params.image_initial_scale, interpolation=cv.INTER_CUBIC)
            power = 0

            image_scores = []
            image_detections = []

            descriptors = []
            yellow_pixels = []
            image_coords = []

            while min(img.shape[0], img.shape[1]) > self.params.dim_window:
                for fx, fy in self.params.window_scales:
                    distorted_img = cv.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv.INTER_AREA)
                    if min(distorted_img.shape[0], distorted_img.shape[1]) < self.params.dim_window:
                        break
                    distorted_color_img = cv.resize(color_img, (0, 0), fx=fx, fy=fy, interpolation=cv.INTER_AREA)
                    hog_descriptor = hog(distorted_img,
                                         pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                         cells_per_block=self.params.cells_per_block, feature_vector=False)
                    whole_masked_image = cv.inRange(cv.cvtColor(distorted_color_img, cv.COLOR_BGR2HSV),
                                                    self.params.hsv_color1, self.params.hsv_color2)
                    num_cols = distorted_img.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = distorted_img.shape[0] // self.params.dim_hog_cell - 1
                    num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - (
                            self.params.cells_per_block[0] - 1)

                    for y in range(0, num_rows - num_cell_in_template, self.params.step_between_windows):
                        for x in range(0, num_cols - num_cell_in_template, self.params.step_between_windows):
                            x_index_image = x * self.params.dim_hog_cell
                            y_index_image = y * self.params.dim_hog_cell
                            descr = hog_descriptor[y: y + num_cell_in_template, x: x + num_cell_in_template].flatten()
                            window_masked_img = whole_masked_image[
                                                y_index_image: y_index_image + self.params.dim_window,
                                                x_index_image:x_index_image + self.params.dim_window]
                            counter_yellow = (window_masked_img == 255).sum()
                            descriptors.append(descr)
                            yellow_pixels.append(counter_yellow)
                            actual_zoom = 1 / (
                                    self.params.image_initial_scale * (self.params.image_minimize_scale ** power))
                            x_min = int((x * self.params.dim_hog_cell / fx) * actual_zoom)
                            y_min = int((y * self.params.dim_hog_cell / fy) * actual_zoom)
                            x_max = int(
                                ((x * self.params.dim_hog_cell + self.params.dim_window) / fx) * actual_zoom)
                            y_max = int(
                                ((y * self.params.dim_hog_cell + self.params.dim_window) / fy) * actual_zoom)
                            image_coords.append([x_min, y_min, x_max, y_max])
                img = cv.resize(img, (0, 0), fx=self.params.image_minimize_scale, fy=self.params.image_minimize_scale,
                                interpolation=cv.INTER_AREA)
                color_img = cv.resize(color_img, (0, 0), fx=self.params.image_minimize_scale,
                                      fy=self.params.image_minimize_scale,
                                      interpolation=cv.INTER_AREA)
                power += 1

            descriptors = np.array(descriptors)
            yellow_pixels = np.array(yellow_pixels)
            predicted_scores = self.neural_model.predict(descriptors)
            for index in range(len(descriptors)):
                score = predicted_scores[index][0]
                counter_yellow = yellow_pixels[index]
                if score > self.params.threshold and counter_yellow > (
                        self.params.dim_window ** 2) * self.params.yellow_percentage:
                    image_detections.append(image_coords[index])
                    image_scores.append(score)

            print(len(image_detections))
            print(image_scores)
            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores),
                                                                              original_image.shape)
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_file_name = ntpath.basename(test_files[i])
                image_names = [short_file_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        print(ground_truth_file_names)
        ground_truth_detections = np.array(ground_truth_file[:, 1:-1], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
