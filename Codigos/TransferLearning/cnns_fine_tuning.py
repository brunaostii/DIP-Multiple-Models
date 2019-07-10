#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Bruna Almeida Osti - 1829009"

import tensorflow as tf
import cv2
import keras
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

#classe de arquiteturas cnns com fine-tuning e imagenet
class Arquiteturas:
    

    def __init__(self, dataset_path, classes, train_filenames, test_filenames):
        self.dataset_path = dataset_path
        self.classes = classes
        self.train_filenames = train_filenames
        self.test_filenames = test_filenames

    def Xception(self):
        print("Xception")
        model_cnn = tf.keras.applications.Xception(weights='imagenet', include_top = False) #carregando o modelo sem a camada densa
        x = model_cnn.output #pegando a saida da rede
        x =tf.keras.layers.GlobalAveragePooling2D()(x) #adicionando camada de pooling
        x = tf.keras.layers.Dropout(0.5)(x) #adicionando camada de dropout
        x = tf.keras.layers.Dense(512, activation='relu')(x) #adicionando camada densa com ativação relu
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x) #adicionando camada densa com ativação softmax
        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions) #criando um modelo utilizável
        model.summary() #mostrando todas as camadas existentes no modelo
        
        
        for layer in model_cnn.layers[:-5]:
            layer.trainable = False #colocando as camadas como não treinaveis, exceto a ultima 
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        

        model.save('Resnet.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test
    
    def VGG16(self):
        print("VGG16")
        
        model_cnn = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        model_cnn.summary()
        model = tf.keras.Sequential()
        
        for layer in model_cnn.layers:
            model.add(layer)
            
        model.layers.pop()
        
        for layer in model.layers:
            layer.trainable = False


        model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
        model.add(tf.keras.layers.Dense(len(self.classes), activation='softmax'))
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])
                      
        
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1)
         

        model.save('VGG16.h5')
        
        modelo_treinado = tf.keras.models.Model(inputs = model.input, outputs = model.output)
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        model_cnn.summary()
        model = tf.keras.Sequential()
        
        for layer in model_cnn.layers:
            model.add(layer)
            
        model.layers.pop()
        
        for layer in model.layers:
            layer.trainable = False


        model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
        model.add(tf.keras.layers.Dense(len(self.classes), activation='softmax'))
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                'plankton_dataset/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                'plankton_dataset/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])             
                      
        
        
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1)
         
       
        model.save('VGG19.h5')
        

        modelo_treinado =tf.keras.models.Model(inputs = model_cnn.input,outputs = model)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        import tensorflow as tf
        model_cnn = tf.keras.applications.ResNet50(weights='imagenet',include_top = False)
       
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
        
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        
        
        model.save('Resnet.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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

        
        model_cnn = keras.applications.keras_applications.resnet_v2.ResNet50V2(include_top = False , weights = 'imagenet' ,
                                     backend = keras . backend , layers = keras . layers , models = keras . models , utils = keras . utils )

        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
        
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        
       
        model.save('Resnetv2.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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

        model_cnn =  keras.applications.keras_applications.resnext.ResNeXt50(include_top = False , weights = 'imagenet' ,
                                     backend = keras . backend , layers = keras . layers , models = keras . models , utils = keras . utils )
        
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
        
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        
       
        model.save('Resnext.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        model_cnn = tf.keras.applications.InceptionV3(weights='imagenet', include_top = False)
        
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
        
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        

        model.save('InceptionV3.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        model_cnn = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top = False)
        
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        
        model.save('InceptionResNetV2.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        model_cnn = tf.keras.applications.MobileNet(weights='imagenet', include_top = False)
        
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        

        model.save('MobileNet.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        model_cnn = tf.keras.applications.DenseNet121(weights='imagenet', include_top = False)
        
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        
       
        model.save('DenseNet.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
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
        model_cnn = tf.keras.applications.NASNetMobile(weights='imagenet', include_top = False)
        
        x = model_cnn.output
        x =tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(len(self.classes), activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = model_cnn.input,outputs = predictions)
        model.summary()
        
        
        for layer in model_cnn.layers[:-4]:
            layer.trainable = False
        
        
        train_batchsize = 10
        val_batchsize = 1
        
        train_datagen = ImageDataGenerator(rescale=1./255)
 
        validation_datagen = ImageDataGenerator(rescale=1./255)
 
        train_generator = train_datagen.flow_from_directory(
                self.dataset_path +'/train',
                target_size=(224, 224),
                batch_size=train_batchsize,
                class_mode='categorical')
 
        validation_generator = validation_datagen.flow_from_directory(
                self.dataset_path +'/test',
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)

        model.compile(tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                                 metrics=['accuracy'])              
                      
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
       
        model.fit_generator(
              train_generator,
              steps_per_epoch=train_generator.samples/train_generator.batch_size ,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples/validation_generator.batch_size,
              verbose=1,
              callbacks = callbacks)
        
       
        model.save('NASNet.h5')
        
        modelo_treinado = model.output
        modelo_treinado =tf.keras.models.Model(inputs = model.input,outputs = modelo_treinado)
        
        
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
            
                features = modelo_treinado.predict(x)
                
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
            
                features = modelo_treinado.predict(x)
                
                X_test.append(features)
                y_test.append(i)
    
        X_train = np.asarray(X_train)
        X_train = X_train.reshape(-1, len(X_train[0][0]))
        X_test = np.asarray(X_test)
        X_test = X_test.reshape(-1, len(X_test[0][0]))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
                
        return X_train, y_train, X_test, y_test

