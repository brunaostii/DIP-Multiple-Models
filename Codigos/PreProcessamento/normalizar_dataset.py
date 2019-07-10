#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Bruna Almeida Osti - 1829009"

dataset_path = 'full_dataset/' #caminho da pasta

import os
import numpy as np
import cv2

os.mkdir('eye') #fazer uma pasta

classes = np.unique(np.asarray(os.listdir(str(dataset_path)))) #listar as pastas    
filenames = []
test_filenames = []

for i in range(len(classes)):
    os.mkdir('eye/'+ classes[i])
    filenames.append(np.unique(np.asarray(os.listdir(str(dataset_path +\
                                                               classes[i])))))
        
 
for i in range(len(classes)):
    for j in range(len(filenames[i])):
        print(filenames[i][j])
        image = cv2.imread(dataset_path + classes[i] + '/' + filenames[i][j])
        gray_image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) #normalizar imagem
        cv2.imwrite('eye/'+ classes[i] + '/' + filenames[i][j],gray_image) #salvar
 
