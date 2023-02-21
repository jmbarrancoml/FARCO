import cv2, os, random, math, shutil
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
def predict(frame: cv2.Mat, model, threshold: float = 0.5):

    predicted_class = False

    predictions = model.predict(frame)
    print("Predictions", predictions)

    probability = np.amax(predictions)[0]

    if probability >= threshold:
        predicted_class = True
    
    return predicted_class, probability

def getFold(n, humans_folder_path, dogs_folder_path, target_train_path, target_test_path, train_size, test_size, clean_before: bool = False, rnd = None):
    if rnd is None:
        rnd = random.Random()
    if clean_before:
        if os.path.exists(target_train_path):
            print(f'[CLEAN BEFORE] Deleting train folder: {target_train_path}...')
            shutil.rmtree(target_train_path)
        if os.path.exists(target_test_path):
            print(f'[CLEAN BEFORE] Deleting test folder: {target_test_path}...')
            shutil.rmtree(target_test_path)
        
    num_train = math.floor(math.floor(n/2) * train_size)
    num_test = math.ceil(math.floor(n/2) * test_size)
    
    humans_folders = []
    # Human folder
    for filename in os.listdir(humans_folder_path):
        humans_folders.append(filename)
        print(f'[HUMANS] {len(humans_folders)} folders found')
        
    dogs_folders = []
    # Dog folder
    for filename in os.listdir(dogs_folder_path):
        dogs_folders.append(filename)
        print(f'[DOGS] {len(dogs_folders)} folders found')
    
    selected_human_images_train = []
    selected_dog_images_train = []
    count_train = 0
    while count_train != num_train:
        print(f'Iteration {count_train} out of {num_train}')
        human_folder_chosen = rnd.choice(humans_folders)
        dogs_folder_chosen = rnd.choice(dogs_folders)
        
        humans_images = []
        dogs_images = []
        
        # Human folder
        for filename in os.listdir(humans_folder_path + '/' + human_folder_chosen):
            humans_images.append(humans_folder_path + '/' + human_folder_chosen + '/' + filename)
        selected_human_images_train.append(rnd.choice(humans_images))
        
        # Dog folder
        for filename in os.listdir(dogs_folder_path + '/' + dogs_folder_chosen):
            dogs_images.append(dogs_folder_path + '/' + dogs_folder_chosen + '/' + filename)
        selected_dog_images_train.append(rnd.choice(dogs_images))
        
        count_train += 1
        
    selected_human_images_test = []
    selected_dog_images_test = []
    count_test = 0
    while count_test != num_test:
        print(f'Iteration {count_test} out of {num_test}')
        human_folder_chosen = rnd.choice(humans_folders)
        dogs_folder_chosen = rnd.choice(dogs_folders)
        
        humans_images = []
        dogs_images = []
        
        # Human folder
        for filename in os.listdir(humans_folder_path + '/' + human_folder_chosen):
            humans_images.append(humans_folder_path + '/' + human_folder_chosen + '/' + filename)
        selected_human_images_test.append(rnd.choice(humans_images))
        
        # Dog folder
        for filename in os.listdir(dogs_folder_path + '/' + dogs_folder_chosen):
            dogs_images.append(dogs_folder_path + '/' + dogs_folder_chosen + '/' + filename)
        selected_dog_images_test.append(rnd.choice(dogs_images))
        
        count_test += 1

    print('\n\t###### RESULTS ######')
    print(f'TRAIN:')
    print(f'\t - Num of humans for train: {len(selected_human_images_train)}')
    print(f'\t - Num of dogs for train: {len(selected_dog_images_train)}')
    print(f'TEST:')
    print(f'\t - Num of humans for test: {len(selected_human_images_test)}')
    print(f'\t - Num of dogs for test: {len(selected_dog_images_test)}')
    

    # #### TRAIN #### #
    # Copiar un archivo de origen a destino
    if not os.path.exists(target_train_path):
        print('Train path does not exists. Creating folder...')
        os.mkdir(target_train_path)
    human_temp_path = target_train_path + '/0humans'
    dog_temp_path = target_train_path + '/1dogs'
    if not os.path.exists(human_temp_path):
        print('Train human path does not exists. Creating folder...')
        os.mkdir(human_temp_path)
    if not os.path.exists(dog_temp_path):
        print('Train dog path does not exists. Creating folder...')
        os.mkdir(dog_temp_path)
    
    for image in selected_human_images_train:
        shutil.copy(image, human_temp_path)
        
    for image in selected_dog_images_train:
        shutil.copy(image, dog_temp_path)

    # #### TEST #### #
    # Copiar un archivo de origen a destino
    if not os.path.exists(target_test_path):
        print('Test path does not exists. Creating folder...')
        os.mkdir(target_test_path)
    human_temp_path = target_test_path + '/0humans'
    dog_temp_path = target_test_path + '/1dogs'
    if not os.path.exists(human_temp_path):
        print('Test human path does not exists. Creating folder...')
        os.mkdir(human_temp_path)
    if not os.path.exists(dog_temp_path):
        print('Test dog path does not exists. Creating folder...')
        os.mkdir(dog_temp_path)
    
    for image in selected_human_images_test:
        shutil.copy(image, human_temp_path)
        
    for image in selected_dog_images_test:
        shutil.copy(image, dog_temp_path)


    
getFold(40, 'dataset/humans_full', 'dataset/dogs_full', 'dataset/train', 'dataset/validation', 0.8, 0.2, clean_before=True)