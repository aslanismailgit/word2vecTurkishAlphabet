import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
import nltk
import pickle
import time

#%%
with open('dictioanry_tr.pickle', 'rb') as f:
    tr_df = pickle.load(f)
#%%



kelime = ["k", "a", "b", "i", "k"]
kal = ["a","o","u","ı"]
inc = ["e","ö","ü","i"]
#%%
def uyum(kelime):
    uy = 0
    match_kal = [s for s in kelime if any(xs in s for xs in kal)]
    match_ince = [s for s in kelime if any(xs in s for xs in inc)]
    if len(match_ince)*len(match_kal)==0:
        print(kelime, "sesli uyumu var")
        uy = 1
    else:
        print(kelime, "sesli uyumu yok")
    return uy
#%%
count = 0
for kelime in tr_df["spaced"]:
    uy = uyum(kelime)
    count=count+uy
count
#%%
uyum_yuzdesi = 100*count/tr_df.shape[0]

#%%
for let in len(uniq_words):
    print(let)
#%%
W=[]
for i,kelime in enumerate(tr_df["spaced"]):
    if "w" in kelime:
        W.append(i)
        print(kelime, "w bulundu")
W
#%% 
Q=[]
for i,kelime in enumerate(tr_df["spaced"]):
    if "q" in kelime:
        Q.append(i)
        print(kelime, "q bulundu")
Q       
