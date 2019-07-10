#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Bruna Almeida Osti - 1829009"

import time

#classe para os classificadores 
class Classificadores:
    
    #pegando os parametros e passando-os para parametros da classe
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier #Importando o modelo
        
        inicio = time.time() #iniciando a contagem  de tempo para treinamento
        classifier = KNeighborsClassifier(n_neighbors=1, n_jobs = -1) #instanciando o KNN com 1 vizinho proximo apenas, e n_jobs serve para rodar em paralelo  
        classifier = classifier.fit(self.X_train, self.y_train) #treinando o classificador
        train_time = time.time() - inicio #terminando a contagem do tempo para treinamento
        
        inicio = time.time() #iniciando a contagem de tempo para teste
        y_pred = classifier.predict(self.X_test) #testando o classificador
        test_time = time.time() - inicio #terminando a contagem de tempo para o teste
                                        
        return y_pred, train_time, test_time #retornando os resultados de predição(teste) e os tempos respectivos de treinamento e teste
    
    def SVM(self):
        from sklearn.svm import LinearSVC #Utilizei o svc -> SVM para utilização como classificador

        inicio = time.time()
        clf = LinearSVC(tol=1e-2) #criterio de tolerancia de parada
        clf = clf.fit(self.X_train, self.y_train) #treinando o classificador
        train_time = time.time() - inicio

        inicio = time.time()
        y_pred = clf.predict(self.X_test) #testando o classificador
        test_time = time.time() - inicio
        
        return y_pred, train_time, test_time
    
    def RandomForest(self):
        from sklearn.ensemble import RandomForestClassifier
        
        inicio = time.time()

        #n_estimators -> quantidade de arvores na floresta, max_depth -> profundidade maxima, 
        #random_state -> semente inicial, não está como aleatória, n_jobs -> processamento paralelo 
        clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                    random_state=0, n_jobs=-1) 
        clf = clf.fit(self.X_train,self.y_train) 
        train_time = time.time() - inicio

        inicio = time.time()
        y_pred = clf.predict(self.X_test)     
        test_time = time.time() - inicio

        return y_pred, train_time, test_time
    
    def J4_8(self):
        from sklearn.tree import DecisionTreeClassifier
        
        inicio = time.time()

        clf = DecisionTreeClassifier(random_state = 0) #instanciando o classificador, estado não aleatório
        clf = clf.fit(self.X_train, self.y_train) #treinando o classificador
        train_time = time.time() - inicio
        
        inicio = time.time()
        y_pred = clf.predict(self.X_test)
        test_time = time.time() - inicio
        
        return y_pred, train_time, test_time
    
    def MLP(self):
        import numpy as np
        from keras.utils import np_utils
#transformar formato para o keras (classe categoricas)
        y_train_hot = np_utils.to_categorical(self.y_train,len(np.unique(self.y_train)))
        
        from keras.models import Sequential 
        from keras.layers import Dense, Dropout

        inicio = time.time()

        model = Sequential()
        model.add(Dense(2048, input_dim=len(self.X_train[0]), activation='relu')) #adicionar camada densa, com ativador relu
        model.add(Dropout(0.8)) #adicionando camada de dropout
        model.add(Dense(1024, activation='relu')) #adicionando camada densa com ativação relu
        model.add(Dropout(0.8)) #adicionando camada de dropout
        model.add(Dense(len(np.unique(self.y_train)), activation='sigmoid')) #adicionando camada densa com ativação sigmoid
 #utilizando a compilação do modelo utilizando a função de perca binary_crossentropy,  o otimizador rmsprop, metrica acuracia
        model.compile(loss='binary_crossentropy', 
                      optimizer='rmsprop',
                      metrics=['accuracy']) 
#treinando o classificador,  20 epocas,  tamanho do batch,  informação sobre o treinamento
        model.fit(self.X_train, y_train_hot, 
                  epochs=20,               
                  batch_size=128,          
                  verbose = 1)             
        train_time = time.time() - inicio

        inicio = time.time()
        y_pred_train = model.predict(self.X_test)
        max_y_pred_train = np.argmax(y_pred_train, axis=1) #tira o dado de forma categorica do keras para a utilização da metrica do sklearn
        test_time = time.time() - inicio

        return max_y_pred_train, train_time, test_time

