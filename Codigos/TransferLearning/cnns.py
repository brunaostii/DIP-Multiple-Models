#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Bruna Almeida Osti - 1829009"

import tensorflow as tf
import cv2
import keras
import numpy as np
from tqdm import tqdm_notebook as tqdm

class Arquiteturas:
    
    #pega os parametros de entrada e salva como parametros da classe
    def __init__(self, dataset_path, classes, train_filenames, test_filenames):
        self.dataset_path = dataset_path
        self.classes = classes
        self.train_filenames = train_filenames
        self.test_filenames = test_filenames

    #Função para a arquitetura xception
    def Xception(self):
        print("Xception")
        model_cnn = tf.keras.applications.Xception(weights='imagenet') #seleciona o modelo com os pesos da imagenet
        model = model_cnn.output #pega a saida do modelo
        model = tf.keras.layers.Flatten()(model)#adiciona a camada flatten
        
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)#cria um modelo utilizável
        
        X_train = []
        y_train = []
        
        print('Train Files')
        #pega as fotos para predição
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j]) #seleciona o caminho da imagem
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR) #le a imagem
                img = cv2.resize(img, (299, 299)) #muda a dimensão da imagem
                
                x = img/255 #deixar o valor dos pixels da imagem entre 0 e 1
                x = np.expand_dims(x, axis=0) #aumentando mais uma dimensão da imagem por causa do tamanho de entrada da rede cnn
            
                features = model.predict(x) #classificando
                
                X_train.append(features) #salvando os resultados
                y_train.append(i) #salvando os resultados
                
        X_test = []
        y_test = []
        
        print('Test Files')
        #pegando as imagens de teste para a predição
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (299, 299))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train) #transformar em formato de matriz de vetor numpy
        X_train = X_train.reshape(-1, len(X_train[0][0])) #transformar em vetor
        X_test = np.asarray(X_test) #transformar em formato de matriz de vetor numpy
        X_test = X_test.reshape(-1, len(X_test[0][0])) #transformar em vetor
        y_train = np.asarray(y_train) #transformar em vetor
        y_test = np.asarray(y_test) #transformar em vetor
                
        return X_train, y_train, X_test, y_test #retorno do treinamento e do teste
    
    def VGG16(self):
        print("VGG16")
        model_cnn = tf.keras.applications.VGG16(weights='imagenet') #carregando o modelo com os pesos da imagenet
        model = model_cnn.output #Pegando a saida da rede       
        model=tf.keras.models.Model(inputs = model_cnn.input, outputs = model) #tornando em um modelo utilizável
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224)) 
                
                x = img/255
                x = np.expand_dims(x, axis=0)
                
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
    
    def VGG19(self):
        print("VGG19")
        model_cnn = tf.keras.applications.VGG19(weights='imagenet')
        model = model_cnn.output
        
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
        
    
    def ResNet(self):
        print("ResNet")
        model_cnn = tf.keras.applications.ResNet50(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
    
    def ResNetV2(self):
        print("ResNetV2")
        #selecionando o modelo com os pesos da imagenet, e dizendo qual é o backend, layers, models e utils, pois o keras foi feito pra\
        #trabalhar com o keras e o tf.keras, não para o keras_applications diretamente
        model_cnn = keras.applications.keras_applications.resnet_v2.ResNet50V2(include_top = True , weights = 'imagenet' ,\
                                     backend = tf.keras.backend , layers = tf.keras.layers , models = tf.keras.models , utils = tf.keras.utils )
        
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
    
    def ResNeXt(self):
        print("ResNeXt")
        model_cnn = keras.applications.keras_applications.resnext.ResNeXt50(include_top = True , weights = 'imagenet' ,\
                                     backend = tf.keras.backend , layers = tf.keras.layers , models = tf.keras.models , utils = tf.keras.utils )
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
    
    def InceptionV3(self):
        print("InceptionV3")
        model_cnn = tf.keras.applications.InceptionV3(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (299, 299))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (299, 299))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
                
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
    
    def InceptionResNetV2(self):
        print("InceptionResNetV2")
        model_cnn = tf.keras.applications.InceptionResNetV2(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (299, 299))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)

                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (299, 299))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
        
    def MobileNet(self):
        print("MobileNet")
        model_cnn = tf.keras.applications.MobileNet(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
        
    def MobileNetV2(self):
        print("MobileNetV2")
        model_cnn = tf.keras.applications.MobileNetV2(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
        
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        return X_train, y_train, X_test, y_test
        
    def DenseNet(self):
        print("DenseNet")
        model_cnn = tf.keras.applications.DenseNet121(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
                
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        return X_train, y_train, X_test, y_test
        
    def NASNet(self):
        print("NASNet")
        model_cnn = tf.keras.applications.NASNetMobile(weights='imagenet')
        model = model_cnn.output
        model = tf.keras.layers.Flatten()(model)
        model=tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
        X_train = []
        y_train = []
        
        print('Train Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.train_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'train/' + self.classes[i] +\
                               '/' + self.train_filenames[i][j])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_train.append(features)
                y_train.append(i)
                
                
        X_test = []
        y_test = []
        
        print('Test Files')
        for i in tqdm(range(len(self.classes)), 'Class', ascii = True):
            for j in tqdm(range(len(self.test_filenames[i])), 'Files', ascii = True):
                img_path = str(self.dataset_path + 'test/' + self.classes[i] +\
                               '/' + self.test_filenames[i][j])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                
                
                x = img/255
                x = np.expand_dims(x, axis=0)
            
                features = model.predict(x)
                
                X_test.append(features)
                y_test.append(i)
        
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test

