import os
import numpy as np
import cv2 as cv
from collections import defaultdict

root_path = "../antrenare/"

names = ["bart", "homer", "lisa", "marge"]

totalBoxes = defaultdict(lambda: [])
characters = []
nb_examples = 0

def intersection(current_box, detection_box):
    x_a = max(current_box[0], detection_box[0])
    y_a = max(current_box[1], detection_box[1])
    x_b = min(current_box[2], detection_box[2])
    y_b = min(current_box[3], detection_box[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (current_box[2] - current_box[0] + 1) * (current_box[3] - current_box[1] + 1)
    box_b_area = (detection_box[2] - detection_box[0] + 1) * (detection_box[3] - detection_box[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


if __name__ == "__main__":
    for name in names:
        filename_annotations = root_path + name + ".txt"
        f = open(filename_annotations)
        for line in f:
            a = line.split(os.sep)[-1]
            b = a.split(" ")

            image_name = root_path + name + "/" + b[0]
            bbox = [int(b[1]), int(b[2]), int(b[3]), int(b[4])]
            character = b[5][:-1]

            totalBoxes[image_name].append(bbox)
            characters.append(character)
            nb_examples = nb_examples + 1

    width_hog = 36
    height_hog = 36

    # compute negative examples using 36 X 36 template
    limit = 100000
    room_limit = 20
    generated = 0

    for idx, img_name in enumerate(totalBoxes.keys()):
        print(idx, img_name)
        img = cv.imread(img_name)
        print("img shape")
        print(img.shape)
        if generated > limit:
            break

        num_rows = img.shape[0]
        num_cols = img.shape[1]
        current_room_generated = 0

        for x in range(0, num_cols - width_hog, width_hog):
            for y in range(0, num_rows - height_hog, height_hog):
                if current_room_generated > room_limit:
                    break
                bbox_curent = [x, y, x + width_hog, y + height_hog]

                for detection in totalBoxes[img_name]:
                    if intersection(bbox_curent, detection) > 0.2:
                        continue

                xmin = bbox_curent[0]
                ymin = bbox_curent[1]
                xmax = bbox_curent[2]
                ymax = bbox_curent[3]
                negative_example = img[ymin:ymax, xmin:xmax]
                filename = "../data/negativeExamples/" + str(generated) + ".jpg"
                generated += 1
                current_room_generated += 1
                cv.imwrite(filename, negative_example)

                # bbox_curent = []
                # ok = False
                # step = 0
                # while not ok or step > 20:
                #     ok = True
                #     x = np.random.randint(low=0, high=num_cols - width_hog)
                #     y = np.random.randint(low=0, high=num_rows - height_hog)
                #
                #     bbox_curent = [x, y, x + width_hog, y + height_hog]
                #
                #     for detection in totalBoxes[img_name]:
                #         if intersection(bbox_curent, detection) > 0.3:
                #             ok = False
                #     step += 1
                #
                # xmin = bbox_curent[0]
                # ymin = bbox_curent[1]
                # xmax = bbox_curent[2]
                # ymax = bbox_curent[3]
                # negative_example = img[ymin:ymax, xmin:xmax]
                # filename = "../data/negativeExamples/" + str(idx * number_negatives_per_image + i) + ".jpg"
                # cv.imwrite(filename, negative_example)
