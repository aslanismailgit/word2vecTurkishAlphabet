
import pandas as pd
import numpy as np
import pickle
import time
from harf_tf_func import *
import tensorflow as tf
from tensorflow import keras

#%%

fiil_df = pd.read_csv("200fiil.csv")

ekler_df = pd.read_csv("ekler.csv")


#%%
with open('./data/dictioanry_tr.pickle', 'rb') as f:
    tr_df = pickle.load(f)

word_letter = tr_df["spaced"]
#test_df=pd.read_csv("./data/test.csv")
fiil_df = pd.read_csv("./data/200fiil.csv")
ekler_df = pd.read_csv("./data/ekler.csv")
#%%
#53
m="S2_B1_F0_E50_28_0928"
WINDOW_SIZE_BACK = 1
WINDOW_SIZE_FORWARD = 0

#30
#m="S2_B1_F1_E50_27_2359"
#WINDOW_SIZE_BACK = 1
#WINDOW_SIZE_FORWARD = 1

#61
#m="S2_B2_F2_E50_28_0047"
#WINDOW_SIZE_BACK = 2
#WINDOW_SIZE_FORWARD = 2

#35
#m="S2_B3_F0_E50_28_0922"
#WINDOW_SIZE_BACK = 3
#WINDOW_SIZE_FORWARD = 0

#%61
#m="S2_B3_F3_E50_28_0853"
#WINDOW_SIZE_BACK = 3
#WINDOW_SIZE_FORWARD = 3

#61
#m="S2_B3_F3_E50_28_0853"
#WINDOW_SIZE_BACK = 3
#WINDOW_SIZE_FORWARD = 3

#58
m="S100_B3_F3_E50_28_1108"
WINDOW_SIZE_BACK = 3
WINDOW_SIZE_FORWARD = 3

model_name = "./tf_modeller/" + m
loaded_model = tf.keras.models.load_model(model_name)
df_labeled = windowize(word_letter,WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD )
lb, X,Y = labelize(df_labeled)
w2v_df = create_w2v(loaded_model,lb)
df_sim =create_simm(w2v_df)
  
#%%
def get_score(df_sim, kok, ek):
    s1=0
    for k in kok:
       sim1=0
       for j in ek:
           sim1=df_sim.loc[k, j]
           s1=s1+sim1
    return s1      
#%%
def metric(df_sim, kok, ek,cevap):
    score = []
    sonuc = 0
    for ek in ekler:
        s = get_score(df_sim, kok, ek)
        score.append(s)
        print(kok,ek,s)
    e = ekler[np.where(score==max(score))[0][0]]
    if e==cevap:
        sonuc=1
#        print(i,"---> sonuç doğru",kok+e)
    else:
        print("-------------------")
#        print(i,"---> sonuç yanlış",kok+e)
    return sonuc,e
#%%
count=0
r=fiil_df.shape[0]
ekler=ekler_df["diliGecmis"]
for i in range(r):
    kok = fiil_df["fiil"][i]
    kok=kok.replace(" ","")
    cevap = fiil_df["cevap"][i]
    print("===================")
#    print(kok,ekler,cevap)
    print("-------------------")
    sonuc,e = metric(df_sim, kok, ekler,cevap)
    count=count+sonuc
100*count/r
#    print(son[i])

#data=([KOK,CEVAP,EKLER])
#df_test = pd.DataFrame(data, index=["kok","cevap","ekler"])
#df_test = df_test.transpose()
#df_test.to_csv("test.csv")
