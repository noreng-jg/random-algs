#refs:http://www.bloodpressureuk.org/BloodPressureandyou/Thebasics/Bloodpressurechart
#https://www.tuasaude.com/imc/

imc=lambda peso,altura: (peso/(altura**2))*1e4

grupos_pressao=[0,1]
classificação=['normal','hipertensão']

grupo_imc=[0,1,2,3]
class_imc=['abaixo do peso','normal','acima do peso','obesidade']

grupo_imc=dict(zip(grupo_imc,class_imc))
grupos_pressao=dict(zip(grupos_pressao,classificação))

def situacao_pressao(numero):
    return grupos[numero]

def situacao_imc(numero):
    return grupo_imc[numero]

def entre(pd,ps,lid,lsd,lis,lss):
    return (pd>=lid and pd<=lsd) and (ps>=lis and ps<=lss)

def categoriza_imc(imc):
    if imc<=18.4:
        return 0
    elif imc>18.4 and imc<=24.9:
        return 1
    elif imc>24.9 and imc<=29.9:
        return 2
    else:
        return 3

def categoriza_pressao(diastolica,sistolica):
    if diastolica<89 and sistolica<139:
        return 0 #normal 
    else:
        return 1 #hipertenso

#print(situacao_pressao(categoriza_pressao(90,159))) teste

import pandas as pd
dados=pd.read_excel('../datasets/cardio_train.xlsx')

#selecionando arrays numpy do dataset
valores_pressao=dados.loc[:,['ap_lo', 'ap_hi']].values
valores_imc=dados.loc[:,['weight', 'height']].values

#classificando novo array numpy com base nos anteriores
[categoriza_pressao(*tuple(i)) for i in valores_pressao]
[categoriza_imc(imc(*tuple(valor))) for valor in valores_imc]

#Ajustes no Dataframe
dados=dados.drop(['ap_hi','ap_lo','weight','height'],axis=1)
serie_imc=pd.Series([categoriza_imc(imc(*tuple(valor))) for valor in valores_imc])
serie_pressao=pd.Series([categoriza_pressao(*tuple(i)) for i in valores_pressao])
dados=dados.drop(['ap_hi','ap_lo','weight','height'],axis=1)
dados['imc'],dados['pressao']=serie_imc,serie_pressao
cardio_serie=dados['cardio'];dados=dados.drop('cardio',axis=1);dados['cardio']=cardio_serie


##Carregando do dataset 
path='../datasets/new_cardio_train.xlsx'

#Classificação de acordo com a idade
classificacao_idade=['criancas/adolescentes','adulto/meia-idade','velhice']
grupo_idade[0,1,2]
grupo_idade=dict(zip(grupo_idade, classificacao_idade))

def situacao_idade(numero):
    return grupo_idade[numero]

anos=lambda dias:dias/365

def categoriza_idade(idade):
    #o dataset nao inclui menores de 25
    if idade>25 and idade<40:
        return 0
    elif idade>=40 and idade<55:
        return 1
    else:
        return 2

 serie_idade=[categoriza_idade(anos(idade)) for idade in vetor_idade]       

#Exportando novo Dataset com modificações
dados.to_excel(r'new_cardio_train.xlsx')
data_cardio=data_cardio.iloc[:,2:]
ata_cardio.loc[:,1]=pd.Series(serie_idade)