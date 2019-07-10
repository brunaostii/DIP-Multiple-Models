# -*- coding: utf-8 -*-
"Bruna Almeida Osti - 1829009"

dataset_path = '' #variavel global com o caminho da pasta da base de dados

import os
import numpy as np
import sys
from cnns import Arquiteturas as cnn #Classe de Arquiteturas
from classificator import Classificadores #Classe de Classificadores
from metricas import Metricas #Classe de métricas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Evitar msgs frequentes do tensorflow


#Função para pegar o nome dos arquivos de treinamento e teste direto da pasta
def names_classes():
    classes = np.unique(np.asarray(os.listdir(str(dataset_path + 'train'))))    
    train_filenames = []
    test_filenames = []
    
    for i in range(len(classes)):
        train_filenames.append(np.unique(np.asarray(os.listdir(str(dataset_path +\
                                                                   'train/' +\
                                                                   classes[i])))))
        
        test_filenames.append(np.unique(np.asarray(os.listdir(str(dataset_path +\
                                                                   'test/' +\
                                                                   classes[i])))))
    
    return train_filenames, test_filenames, classes


#Função para selecionar a arquitetura e os classificadores 
def main(arquitetura, classificador, output):
    train_filenames, test_filenames, classes = names_classes() #pega os nomes dos arquivos de treinamento e teste 
    cnns = cnn(dataset_path, classes, train_filenames, test_filenames) #instancia a classe das arquiteturas
    metrica = Metricas() #instancia a classe das metricas
    classificador_teste = 0 #variavel para evitar a sobreposição do txt de saída
    
    if(arquitetura == "Xception"):

        X_train, y_train, X_test, y_test = cnns.Xception() #chamar a arquitetura escolhida
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'Xception', classificador_teste) #chamar o classificador escolhido
        classificador_teste += 1 
    
    if(arquitetura == "VGG16" ):
        X_train, y_train, X_test, y_test = cnns.VGG16()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'VGG16', classificador_teste)
        classificador_teste += 1
       
    if(arquitetura == "VGG19" ):
        X_train, y_train, X_test, y_test = cnns.VGG19()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'VGG19', classificador_teste)
        classificador_teste += 1
            
    if(arquitetura == "ResNet" ):
        X_train, y_train, X_test, y_test = cnns.ResNet()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'ResNet', classificador_teste)
        classificador_teste += 1

    if(arquitetura == "ResNetV2"):
        X_train, y_train, X_test, y_test = cnns.ResNetV2()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'ResNetV2', classificador_teste)
        classificador_teste += 1

    if(arquitetura == "ResNeXt" ):
        X_train, y_train, X_test, y_test = cnns.ResNeXt()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'ResNeXt', classificador_teste)
        classificador_teste += 1

    if(arquitetura == "InceptionV3"):
        X_train, y_train, X_test, y_test = cnns.InceptionV3()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'InceptionV3', classificador_teste)
        classificador_teste += 1 
            
    if(arquitetura == "InceptionResNetV2" ):
        X_train, y_train, X_test, y_test = cnns.InceptionResNetV2()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'InceptionResNetV2', classificador_teste)
        classificador_teste += 1
            
    if(arquitetura == "MobileNet" ):
        X_train, y_train, X_test, y_test = cnns.MobileNet()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'MobileNet', classificador_teste)
        classificador_teste += 1    
            
    if(arquitetura == "MobileNetV2" ):
        X_train, y_train, X_test, y_test = cnns.MobileNetV2()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'MobileNetV2', classificador_teste)
        classificador_teste += 1
            
    if(arquitetura == "DenseNet"):
        X_train, y_train, X_test, y_test = cnns.DenseNet()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'DenseNet', classificador_teste)
        classificador_teste +=1
        
    if(arquitetura == "NASNet"):
        X_train, y_train, X_test, y_test = cnns.NASNet()
        classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, 'NASNet', classificador_teste)
        classificador_teste += 1

def classificadores(X_train, y_train, X_test, classificador, metrica, y_test, output, arquitetura, classificador_teste):
        
        clas = Classificadores(X_train, y_train, X_test) #instancia a classe dos classificadores
        
        if(classificador == "KNN" or classificador == ''): 
            print('KNN')
            y_pred, time_train, time_test = clas.KNN() #pega as instancias de saída
            acc = metrica.acuracia(y_test, y_pred) #gera a metrica de acuracia
            matrix = metrica.matrix_de_confusao(y_test, y_pred) #gera a matriz de confusão
            precisao = metrica.precisao(y_test, y_pred) #gera a metrica de precisão
            revocacao = metrica.revocacao(y_test, y_pred) #gera a metrica de revocação
            classificador_teste +=2
            print('Tempo: \n Treinamento:  ', time_train, '\n Teste: ', time_test) #imprime o tempo de teste
            resultados(output, arquitetura, 'KNN', acc, matrix, precisao, revocacao, \
               time_train, time_test, classificador_teste) #gera o txt de resultados 
            
        if(classificador == "SVM" or classificador == ''):
            print('SVM')
            y_pred, time_train, time_test = clas.SVM()
            acc = metrica.acuracia(y_test, y_pred)
            matrix = metrica.matrix_de_confusao(y_test, y_pred)
            precisao = metrica.precisao(y_test, y_pred)
            revocacao = metrica.revocacao(y_test, y_pred)
            classificador_teste +=3
            print('Tempo: \n Treinamento:  ', time_train, '\n Teste: ', time_test)
            resultados(output, arquitetura, 'SVM', acc, matrix, precisao, revocacao, \
               time_train, time_test, classificador_teste)
        
        if(classificador == "RandomForest" or classificador == ''):   
            print('RandomForest')
            y_pred, time_train, time_test = clas.RandomForest()
            acc = metrica.acuracia(y_test, y_pred)
            matrix = metrica.matrix_de_confusao(y_test, y_pred)
            precisao = metrica.precisao(y_test, y_pred)
            revocacao = metrica.revocacao(y_test, y_pred)
            classificador_teste += 5 
            print('Tempo: \n Treinamento:  ', time_train, '\n Teste: ', time_test)
            resultados(output, arquitetura, 'RandomForest', acc, matrix, precisao, revocacao, \
               time_train, time_test, classificador_teste)
            
        if(classificador == "J4_8" or classificador == ''):
            print('J4_8')
            y_pred, time_train, time_test = clas.J4_8()
            acc = metrica.acuracia(y_test, y_pred)
            matrix = metrica.matrix_de_confusao(y_test, y_pred)
            precisao = metrica.precisao(y_test, y_pred)
            revocacao = metrica.revocacao(y_test, y_pred)
            classificador_teste += 7 
            print('Tempo: \n Treinamento:  ', time_train, '\n Teste: ', time_test)
            resultados(output, arquitetura, 'J4_8', acc, matrix, precisao, revocacao, \
               time_train, time_test, classificador_teste)
            
        if(classificador == "MLP" or classificador == ''):
            print('MLP')
            y_pred, time_train, time_test = clas.MLP()
            acc = metrica.acuracia(y_test, y_pred)
            matrix = metrica.matrix_de_confusao(y_test, y_pred)
            precisao = metrica.precisao(y_test, y_pred)
            revocacao = metrica.revocacao(y_test, y_pred)
            classificador_teste += 9
            print('Tempo: \n Treinamento:  ', time_train, '\n Teste: ', time_test)
            resultados(output, arquitetura, 'MLP', acc, matrix, precisao, revocacao, \
               time_train, time_test, classificador_teste)
            
def resultados(output, arquitetura, classificador, acc, matriz, precisao, revocacao, \
               time_train, time_test, classificador_teste):
    
    filename = open(output + str(classificador_teste) + '.txt','w+') #cria o arquivo txt    
    filename.write(arquitetura + ' - ' + classificador + '\n\n') #adiciona o nome da arquitetura e do classificador
    filename.write('Tempo de Treino: ' + str(time_train) + '\n') #adiciona o tempo de treinamento
    filename.write('Tempo de Teste: ' + str(time_test) + '\n') #adiciona o tempo de teste
    filename.write('Acurácia: ' + str(acc) + '\n') #adiciona a acuracia
    filename.write('Precisao: ' +str(precisao) + '\n') #adiciona a precisão
    filename.write('Revocao: ' + str(revocacao) + '\n') #adiciona a revocação
    filename.write('Matriz de Confusão: ' + '\n' + str(matriz)) #adiciona a matriz de confusão
    filename.close() #salva e fecha o arquivo txt
    
        
            

if __name__ == "__main__":
    
    param = sys.argv[1:] #pega os parametros de entrada
    arquitetura = ""
    output = ""
    classif = ""
    
    #compara a entrada para pegar os parametros
    if(param[0] == "-h"):
        arquitetura = param[1]
        
    if(param[0] == "-o"):
        output = param[1]
        
        if(param[2] == "-d"):
            dataset_path = param[3]
    
    if(param[2] == "-o"):
        output = param[3]
        
        if(param[4] == "-d"):
            dataset_path = param[5]
            
        
    elif(param[2] == "-c"):
        classif = param[3]
        
        if(param[4] == "-o"):
            output = param[5]
        
        if(param[6] == "-d"):
            dataset_path = param[7]
    
    main(arquitetura, classif, output)
    
    

