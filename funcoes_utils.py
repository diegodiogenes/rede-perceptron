# Arquivo reunindo funções auxiliares para a implementação da rede neural perceptron
import numpy as np
import csv

def carregar_dados(filename):
    """ Essa função tem como objetivo orgarnizar 
    o dataset para o treino da rede. """
    
    with open(filename) as dataset:
        # lemos o csv e colocamos em um numpy array
        dados = np.array(list(csv.reader(dataset)))
        # pegamos os dois tipos de classes de flores do csv que estão na última coluna
        classes = np.array(list(set(dados[1:,-1]))) #utilizamos set para não termos valores repetidos
        #criamos um array para armazenar as features presentes no dataset
        features = np.zeros((len(dados)-1,len(dados[0])-1)) #retiramos 1 por causa do cabeçalho
        #criamos um array para armazenar os resultados na saída (0 ou 1) no nosso caso
        saida_y = np.zeros(len(dados)-1)
        
        # realizamos um for para percorrer a matriz de dados, começando do 1, excluindo o cabeçalho
        for i in range(1, len(dados)):
            features[i-1] = dados[i,:-1] # armazenamos apenas as features, excluindo as classes na última coluna    
            # criamos um for para atribuir valores floats ou inteiros para os resultados de saída
            for j in range(len(classes)):
                if classes[j] in dados[i]: # verifica qual classe está contida na linha
                    saida_y[i-1] = j

    return features, saida_y
