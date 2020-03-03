import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from harf_gen_func import *

#%%
with open('./data/dictioanry_tr.pickle', 'rb') as f:
    tr_df = pickle.load(f)

word_letter = tr_df["spaced"]
#test_df=pd.read_csv("./data/test.csv")
fiil_df = pd.read_csv("./data/200fiil.csv")
ekler_df = pd.read_csv("./data/ekler.csv")
  
#%%
def get_score(df_sim, kok, ek):
#    df_sim.set_index('word')
    s1=0
    for k in kok:
       sim1=0
       for j in ek:
           sim1=df_sim.loc[k, j]
           print(k,j,sim1)
           s1=s1+sim1
    return s1      
#%%
def metric(df_sim, kok, ek,cevap):
#    df_sim.set_index('word')
    score = []
    sonuc = 0
    for ek in ekler:
        s = get_score(df_sim, kok, ek)
        score.append(s)
#        print(kok,ek,s)
    e = ekler[np.where(score==max(score))[0][0]]
    if e==cevap:
        sonuc=1
#        print(i,"---> sonuç doğru",kok+e)
#    else:
#        print("-------------------")
#        print(i,"---> sonuç yanlış",kok+e)
#    print ("sonuc--->",kok+e)
    return sonuc,e
#%%
def get_result(file,df_sim):
    df_sim=pd.read_csv("./gensim_models/" + file)
    df_sim=df_sim.set_index('word')
    
    count=0
    r=fiil_df.shape[0]
    ekler=ekler_df["diliGecmis"]
    for i in range(r):
        kok = fiil_df["fiil"][i]
        kok=kok[-1]
        kok=kok.replace(" ","")
        cevap = fiil_df["cevap"][i]
#        print("===================")
    #    print(kok,ekler,cevap)
#        print("-------------------")
        sonuc,e = metric(df_sim, kok, ekler,cevap)
        count=count+sonuc
    accu=100*count/r
#    print(accu)
    return accu
#%%
import os
files = os.listdir('./gensim_models/')
for file in files:
    if "sim" in file:
        accu = get_result(file,df_sim)
        print (file,"----> %.2f" %accu)

#%%
#w2v=pd.read_csv("./gensim_models/" + "GEN_S2_BF1_E100_29_2126w2v.csv")

df_sim=pd.read_csv("./gensim_models/" + "GEN_S2_BF1_E100_29_2126sim.csv")
df_sim=df_sim.set_index('word')
#%%
kok="ye"
x=[]
y=[]
s=[]
#bars = ('A', 'B', 'C', 'D', 'E')


for ek in ekler:
    ek=ek[0]
    for k in kok:
       
       for j in ek:
           x.append(k)
           y.append(j)
           s.append(df_sim.loc[k, j])
           print(k,j,df_sim.loc[k, j])

#%%
kok="gel"
k=[k for k in kok]


#%%
from nltk.tokenize import word_tokenize
import nltk
txt = ekler.str.lower().str.cat()
txt_let = [l for l in (txt)]
" ".join(txt_let)
words_except_stop_dist = nltk.FreqDist(w for w in txt_let) 
uniq_words  = {uw for uw in words_except_stop_dist}
uniq_words
#%%
df=df_sim.loc[k,uniq_words]

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap(df, annot=False)

#%%
file="GEN_S2_BF3_E100_29_2127"
df_sim=pd.read_csv("./gensim_models/" + file + "sim.csv")
df_sim=df_sim.set_index('word')
ekler=ekler_df["diliGecmis"]

w2v_df=pd.read_csv("./gensim_models/" + file+"w2v.csv")
w2v_df=w2v_df.set_index('word')

#%%
from sklearn.metrics.pairwise import cosine_similarity
#%su=[]
kok="git"
su_kok = sum_(kok,w2v_df)

for ek in ekler:
    s1 = sum_(ek,w2v_df)
    v1=np.array(su_kok.values).reshape(1,-1)
    v2=np.array(s1.values).reshape(1,-1)
    sim = cosine_similarity(v1,v2)
#    print(su_kok.values.reshape(1,-1),s1.values.reshape(1,-1))
    print(ek,sim)

#%%
def sum_(kok,w2v_df):
#    kok="gel"
    k=[k for k in kok]
    x=["x1","x2"]
    df=w2v_df.loc[k,x]
    su=df.sum(axis=0)
    return su

