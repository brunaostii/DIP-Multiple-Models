# -*- coding: utf-8 -*-
"Bruna Almeida Osti - 1829009"

import os
import random
from scipy import ndarray
import skimage as sk
from skimage import io

def random_rotation(image_array: ndarray):
    # rotaciona a figura em -25 e 25
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # adiciona ruído a imagem
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # virar a imagem
    return image_array[:, ::-1]


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}
classe = 'Cotton-wool Spots', 'Deep Hemorrhages', 'Drusen', 'Hard Exudates',\
    'Normal Images', 'Red Lesions', 'Superficial Hemorrhages'

for i in range(len(classe)):
    print(classe[i])
    folder_path = 'eye_dataset/train/' + classe[i]
    num_files_desired = 1000
    # pega todas as classes
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    num_generated_files = 0
    while num_generated_files <= num_files_desired:
        #escolhe a imagem randomicamente, abre, e aplica a transformação
        image_path = random.choice(images)
        image_to_transform = sk.io.imread(image_path)
        num_transformations_to_apply = random.randint(1, len(available_transformations))
    
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
    
        new_file_path = 'augmented_eye/' + classe[i] + '/' + str(num_generated_files) + '.jpg'
        
        # Salvar
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1
