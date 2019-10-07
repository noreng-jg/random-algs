import scipy.io #Bibliotecas de Utilização no projeto
import numpy as np
import pandas as pd

data=scipy.io.loadmat('data.mat')
dados_treinamento=pd.DataFrame(data['X'])
dados_treinamento['Y']=data['Y'].T
dados_treinamento.index=range(1,1001,1)#mudar o range posição inteiras
#Nomenclatura da Matrix em função das palavras Ws
dados_treinamento.columns=pd.Index(['W1','W2','W3','W4','W5','Y'], dtype='object')
dados_treinamento.head()

#Embaralhando o Dataset
dados_treinamento=dados_treinamento.sample(frac=1)
grupos=np.split(dados_treinamento.values,10)
[len(ar) for ar in grupos]#10 dados de teste com tamnho 100 cada

amostras,teste=[grupos[1:]],[grupos[0]]#primeira
am=[grupos[:1+i]+grupos[i+2:] for i in range(1,9)]#valore intermediarios
teste.extend([grupos[i] for i in range(1,10)])#exapndir lista de teste
am.append(grupos[:-1])#ultima
amostras.extend(am)#extende a lista
print(len(amostras[0]))#cada amostra é composta de 9 listas

#cria dois arrays que recebem os primeiros valores para treinamento e teste respectivamente
#a,b=np.array(amostras[0]),np.array(grupo[0])
amsfinal,b=amostras[0][0],teste[0]

print(len(amostras))#10 amostras no total
print(type(amostras[0]),type(teste))#Elementos do tipo lista

#Gerar 10 subsets de 900 amostras cada
subsets=[np.vstack([j for j in samp]) for samp in amostras]


#Importando dados 
from naive_bayes import *

#nb=nbayes(subsets[0])
treinamentos=[nbayes(subsets[i]) for i in range(len(subsets))]

for nb in treinamentos:#treina todos os subsets
    nb.treinar()

#abaixo lista de predicoes
pred_list=[nb.predicao(teste[i][:,:-1]) for i,nb in enumerate(treinamentos)]

#abaixo lista de teste
test_list=[(teste[i][:,-1]) for i in range(len(teste))]

#lista das probabilidades
p=[[sum([1 if pred_list[j][i]==test_list[j][i] else 0 for i in range(0,100)])/100] for j in range(len(pred_list))]

#probabilidade média
print(np.mean(p))

print(p)