import cv2, os
import numpy as np


def show_data(image, predicted_class, probability):

    # Creamos un rectangulo para tener un fondo para el texto
    cv2.rectangle(image, (10, 10), (200, 100), (0, 255, 0), 3)

    # Agregamos el texto en el rectángulo
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Class: {predicted_class} | Probability: {probability}', (30, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return image

def run_haar_cascade_classifier(image):

    haar_path = '/content/drive/MyDrive/FARCO/classifiers/haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(haar_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
    return image

def nothing(x=None):
    pass

"""
  Args
    frame (cv2.image): el frame actual a predecir
    threshold (float): el umbral mínimo para ser aceptada la clase positiva (Jaume o perro)
  
  returns
    predicted_class (bool): clase aceptada teniendo en cuenta el umbral. Si es la clase negativa False, si no, True
    probability: probabilidad obtenida para la clase aceptada
"""
def predict(frame: cv2.image, model, threshold: float = 0.5):

    predicted_class = False

    predictions = model.predict(frame)
    print("Predictions", predictions)

    probability = np.amax(predictions)[0]

    if probability >= threshold:
        predicted_class = True
    
    return predicted_class, probability

def getFold(n, humans_folder_path, dogs_folder_path, train_path, test_path, train_size, test_size):
    humans_folders = []
    # Human folder
    for filename in os.listdir(humans_folder_path):
        humans_folders.append(filename)
        
    dogs_folders = []
    # Dog folder
    for filename in os.listdir(dogs_folder_path):
        dogs_folders.append(filename)