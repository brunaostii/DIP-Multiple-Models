#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Bruna Almeida Osti - 1829009"

import sklearn.metrics as met

class Metricas:
    
    def __init__(self):
        print("")

    #Função para calcular a acurácia do modelo
    def acuracia(self,y_true, y_pred): 
        acc = met.accuracy_score(y_true, y_pred) #calculo da acuracia
        print('Acurácia:')
        print(acc*100) #deixa em porcentagem
        return acc*100
    
    #Função para calcular a precisão do modelo
    def precisao(self,y_true, y_pred): 
        acc = met.precision_score(y_true, y_pred, average='macro') #calculo da precisão - macro serve para calcular para todos os rotulos 
        print('Precisão:')
        print(acc*100)
        return acc*100
    
    #Função para calcular a revocação do modelo
    def revocacao(self, y_true, y_pred):
        acc = met.recall_score(y_true, y_pred,  average='macro')#calculo da revocação - macro serve para calcular de todos os rotulos
        print('Revocação:')
        print(acc*100)
        return acc*100
    
    #Função para calcular a matriz de confusão do modelo
    def matrix_de_confusao(self, y_true, y_pred): 
        acc = met.confusion_matrix(y_true, y_pred)#calculo da matriz de confusãp
        print('Matriz de Confusão:')
        print(acc)
        return acc
        
