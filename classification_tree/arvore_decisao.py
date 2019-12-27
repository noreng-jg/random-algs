import numpy as np
import pandas as pd
from  math import log2

class Questao:
    
    def __init__(self, nome_feature, valor):
        self.nome_feature=nome_feature
        self.valor=valor
             

class Noh:
    """
    Define o nó, explicitando por padrão a prob
    das classes para calculo probabiistico
    a partir de um pandas DataFrame de entrada

    """

    def porcentagem_classes(self,array_ultima_coluna):
        """Calcula prob de cada classe em um determinado no"""
        unicos, contador = np.unique(array_ultima_coluna, return_counts=True)
        return dict(zip(unicos, contador/array_ultima_coluna.shape[0]))

    def gini(self):
        """Calcula a impureza de gini"""
        impureza=1
        for prob in self.pks.values():
            impureza-=prob*prob
        return impureza

    def shannon(self):
        """Calcula a impureza de Shannon"""
        impureza=0
        for prob in self.pks.values():
            impureza+=prob*log2(prob)
        return -impureza

    def pred(self):
        return max(self.pks, key=self.pks.get)        

    def __init__(self,dados, sim=None, nao=None, questao=None):#método construtor
        self.pks=self.porcentagem_classes(dados.to_numpy()[:,-1])
        self.gini=self.gini()
        self.shannon=self.shannon()
        self.tamanho=len(dados)
        self.dados=dados
        self.sim=sim
        self.nao=nao
        self.questao=questao
    

def particiona(dados, questao: Questao):
    """Divide a matriz de dados de acordo com uma questão estabelecida,
    retornando respectivamente a particao de dados em resultados positivos e negativos
    em formato de tupla
    """
    return dados[dados[questao.nome_feature]==questao.valor],dados[dados[questao.nome_feature]!=questao.valor] 

def decrescimento_impureza(t:Noh,te:Noh,td:Noh, metodo='gini')->float:
    """Recebe as impurezas no nó e dos filhos retornando a metrica
   de descresimento
    """
    #maxpe, maxpd=max(te.pks.values()), max(td.pks.values())#usa a maxima prob da classe
    if metodo=='shannon':
        return t.shannon-(te.tamanho/t.tamanho)*te.shannon-(td.tamanho/t.tamanho)*td.shannon
    else:
        return t.gini-(te.tamanho/t.tamanho)*te.gini-(td.tamanho/t.tamanho)*td.gini


def acha_melhor_questao(dados):
    """Varredura do set de treinamento, buscando a pergunta que minimiza o decrescimento de impureza"""
    deltai=0
    melhor_questao=None
    for coluna in dados.columns[:-1]:#acha melhor questao do dataset, tirando a coluna da label
        for valor in dados[coluna].unique():#acha melhor questao da feature
            questao=Questao(coluna,valor)
            de,dd=particiona(dados, questao)
            no,ne,nd=Noh(dados),Noh(de),Noh(dd)
            if decrescimento_impureza(no,ne,nd)>deltai:
                melhor_questao=questao
                deltai=decrescimento_impureza(no,ne,nd)
    return melhor_questao

def devo_parar(noh,contador,max_profundidade=6,min_amostras=20):
    """Critérios de parada definidos, mudar de acordo com o desejado"""
    return contador==max_profundidade or noh.tamanho<min_amostras

def cresce_arvore(noh, contador=0):
    """Expansão do nó inicial utilizando as partições sucessivas"""
    if not devo_parar(noh,contador):
        contador+=1
        questao=acha_melhor_questao(noh.dados)
        de,dd=particiona(noh.dados, questao)
        ne,nd=Noh(de),Noh(dd)
        noh.questao=questao
        noh.sim, noh.nao=ne,nd
        cresce_arvore(ne, contador)
        cresce_arvore(nd, contador)


def classificar_linha(linha, noh):
    """Classificação para apenas uma linha do treino, percorrendo a árvore"""
    #linha é do tipo Series, semelhante a dicionario
    if noh.sim==None:#folha
        preds.append(noh.pred())
    else:
        if linha[noh.questao.nome_feature]==noh.questao.valor:#analise do valor da feature do noh, indo para nó filho a dir ou esq.
            classificar_linha(linha, noh.sim)
        else:
            classificar_linha(linha, noh.nao)#avanca
            #definir se ir pra a direita ou esquerda de acordo com o valor da feature que está na questão do nó 

def classifica_tudo(dados, arvore):
    """Varredura de todas as linhas"""
    for i in range(len(dados)):
        classificar_linha(dados.iloc[i,:-1], arvore)
    return preds


#Análise dos 2 datasets
dados_doencas='../datasets/new_cardio_train.xlsx'#Dados de Doencas Cardiacas
dados_spam='../datasets/dados_treinamento.xlsx'#Dados de Spam

dados=dados_doencas

#Separando dados de treino e teste, hold-out 70,30
k=int(len(pd.read_excel(dados))*70/100)
dados_trial=pd.read_excel(dados).iloc[:k,1:]
dados_teste=pd.read_excel(dados).iloc[k:,1:]

#Intanciando elemento árvore
raiz=Noh(dados_trial)
arvore=cresce_arvore(raiz)

preds=[]#lista para predicoes

#Percentual de acerto
valores_reais=dados_teste.values[:,-1]
predicoes=np.array(classifica_tudo(dados_teste,raiz))
print(np.mean(predicoes==valores_reais))

