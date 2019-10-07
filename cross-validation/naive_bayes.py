import pandas as pd
import numpy as np


class nbayes:
    """
    teste de 6 colunas
    Matriz de treinamento: Contém os dados de treinamento e os rótulos
    
    Retorna a matriz de predições do método

    """
    def __init__(self, mt):
    
        self.matriz_treinamento=mt; self.py1=None;self.py0=None; self.ps=None; self.ns=None
    
    
    def treinar(self):#Ainda não está genérico serve apenas pro dataset em questão
        #Cast de DataFrame 
        dados=pd.DataFrame(self.matriz_treinamento) 
        #separar valores de saídas 1 e -1
        xyneg=dados[dados.iloc[:,-1]==-1]
        xypos=dados[dados.iloc[:,-1]==1]
        tamanho=len(dados)
        py1=len(xypos)/tamanho
        py0=len(xyneg)/tamanho
        npos=xypos.iloc[:,:-1].to_numpy().sum() #soma de todas as palavras onde y é +1
        nneg=xyneg.iloc[:,:-1].to_numpy().sum() #soma de todas as palavras onde y é -1
        opws=['w1','w2','w3','w4','w5']#ocorrencias positivas das palavras
        onws=['w1','w2','w3','w4','w5']##..neg ..
        soma_colunas_ypos=list(xypos.iloc[:,:-1].sum().to_numpy())
        soma_colunas_yneg=list(xyneg.iloc[:,:-1].sum().to_numpy())
        opws=dict(zip(opws, soma_colunas_ypos))
        onws=dict(zip(onws, soma_colunas_yneg))
        pwkypos,pwkyneg={},{}#probabilidades ocorrencia pos
        
        pwk=lambda nk,n: (nk+1)/(n+5)#minifuncao LS onde 5 é o n° de features, palavras ou classes

        for key in opws.keys():
            pwkypos[key]=pwk(opws[key],npos)

        for key in onws.keys():
            pwkyneg[key]=pwk(onws[key],nneg)

        x_train=self.matriz_treinamento[:,:-1]

        ps=np.array(list(pwkypos.values()))
        ns=np.array(list(pwkyneg.values()))

        self.py1=py1;self.py0=py0; self.ps=ps; self.ns=ns

    #atributos que serão modelados para intuição dos dados de teste 

    def predicao(self,x_test):
        rotulos=[]
        
        for i in range(len(x_test)):
            l0=x_test[i,:]
            tempp=np.dot(l0,self.ps)*self.py1
            tempn=np.dot(l0,self.ns)*self.py0
            
            #checagem do maior produtório das probs abaixo
            if tempp>tempn:
                rotulos.append(1)
            else:
                rotulos.append(-1)
        
        return np.array(rotulos)    

