from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

import numpy as np
import os, time


def test_cnn():
    dogs_dataset_path = '/content/drive/MyDrive/FARCO/dataset/test/dogs/'
    human_dataset_path = '/content/drive/MyDrive/FARCO/dataset/test/faces/'
    altura_img, anchura_img = 32, 32

    total_correct_predictions = 0
    total_images = 0

    correct_dogs = 0
    correct_humans = 0
    total_dogs = 0
    total_humans = 0

    model_path = ''
    weights_path = ''
    cnn = load_model(model_path)
    cnn.load_weights(weights_path)

    times = list()
    for filename in os.listdir(dogs_dataset_path):
        image = load_img(dogs_dataset_path + filename, target_size = (altura_img, anchura_img))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        start_time = time.time()

        predictions = cnn.predict(image)

        end_time = time.time()

        times.append(end_time - start_time)

        if np.argmax(predictions) == 0:
            total_correct_predictions += 1
            correct_dogs += 1

        total_dogs += 1
        total_images += 1

    for filename in os.listdir(human_dataset_path):
        image = load_img(human_dataset_path + filename, target_size = (altura_img, anchura_img))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        start_time = time.time()

        predictions = cnn.predict(image)

        end_time = time.time()

        times.append(end_time - start_time)

        if np.argmax(predictions) == 1:
            total_correct_predictions += 1
            correct_humans += 1

        total_humans += 1
        total_images += 1
    
    print(f'Mean time {np.mean(times)}')

    print(f'Total accuracy: {(total_correct_predictions / total_images) * 100}%')
    print(f'Dog accuracy: {(correct_dogs / total_dogs) * 100}%')
    print(f'Human accuracy: {(correct_humans / total_humans) * 100}%')
    


if __name__ == '__main__':
    test_cnn()