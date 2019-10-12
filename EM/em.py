from scipy.io import loadmat
r=loadmat('formantdata.mat')
import numpy as np
import random

#Carregando Dados
x_val=r['D']
rotulos=np.array([-1 if random.randint(0,1)==0 else 1 for i in range(449)])


##Gaussiana
def distribuicao(x,mu, cov):
    detcov=np.linalg.det(cov)
    k=(1/((2*np.pi)**(2/2))*np.sqrt(detcov))
    e=np.exp(-.5*(np.dot((x-mu).T,(np.linalg.inv(cov)))).dot((x-mu)))
    return k*e

#Parâmetros Iniciais

media_0=sum([np.dot(rotulos[i],x_val[i]) if rotulos[i]==1 else 0 for i in range(len(rotulos))])/sum([1 if i==1 else 0 for i in rotulos])
media_1=sum([np.dot(rotulos[i],x_val[i]) if rotulos[i]==-1 else 0 for i in range(len(rotulos))])/sum([1 if i==-1 else 0 for i in rotulos])

fi1=(1/len(rotulos))*sum([1 if i==1 else 0 for i in rotulos])
fi2=(1/len(rotulos))*sum([1 if i==-1 else 0 for i in rotulos])
mc1=sum([(x_val[i]-media_0).reshape(2,1)*(x_val[i]-media_0).reshape(2,1).T if rotulos[i]==1 else 0 for i in range(len(rotulos))])/sum([1 if i==1 else 0 for i in rotulos])
mc2=sum([(x_val[i]-media_1).reshape(2,1)*(x_val[i]-media_1).reshape(2,1).T if rotulos[i]==-1 else 0 for i in range(len(rotulos))])/sum([1 if i==-1 else 0 for i in rotulos])


for j in range(45):# Loop do Algoritmo EM
    ric1,ric2=[],[]
    for xi in x_val:
        d1=distribuicao(xi,media_0, mc1)
        d2=distribuicao(xi,media_1, mc2)
        #probs abaixo
        ric1.append((fi1*d1)/(fi1*d1+fi2*d2))
        ric2.append((fi2*d2)/(fi1*d1+fi2*d2))

    fi1=sum(ric1)/len(ric1)
    print(fi1)
    fi2=sum(ric2)/len(ric2)
    print(fi2)
    media_0=sum([np.dot(ric1[i],x_val[i]) for i in range(len(x_val))])/sum(ric1)
    media_1=sum([np.dot(ric2[i],x_val[i]) for i in range(len(x_val))])/sum(ric2)
    mc1=sum([ric1[i]*(x_val[i]-media_1).reshape(2,1)*(x_val[i]-media_1).reshape(2,1).T for i in range(len(x_val))])/sum(ric1)
    mc2=sum([ric2[i]*(x_val[i]-media_0).reshape(2,1)*(x_val[i]-media_0).reshape(2,1).T for i in range(len(x_val))])/sum(ric2)


print(media_0,media_1) 

print(mc1, mc2)
    
