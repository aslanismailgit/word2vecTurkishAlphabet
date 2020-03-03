import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#import seaborn as sns
#import re
from nltk.tokenize import word_tokenize
import nltk
import pickle

#%%
with open("./data/"+'zargan.pkl', 'rb') as f:
    tr = pickle.load(f)
tr_df = pd.DataFrame(list(tr.items()), columns=['word', 'value'])

#%%
with open('dictioanry_tr.pickle', 'rb') as f:
    tr_df = pickle.load(f)

#%%
wq=[]
notwq=[]
for i,kelime in enumerate(tr_df["word"]):
    if ("w" in kelime)| ("q" in kelime):
        wq.append(i)
#        tr = tr_df.drop(tr_df.index[i]).copy
        print(kelime, "bulundu")
    else:
        notwq.append(i)
#tr=tr_df.iloc[notwq]
#%% 
#Q=[]
#for i,kelime in enumerate(tr_df["word"]):
#    if "q" in kelime:
#        Q.append(i)
#        tr_df = tr_df.drop(tr_df.index[i])
#        print(kelime, "q bulundu")
#Q
#
#tr = tr_df.drop(tr_df.index[qw])
   
#%%
tr["word"] = [w for w in tr["word"].str.replace('.', '')] 
tr["word"] = [w for w in tr["word"].str.replace("'", '')] 
#%%
tr["spaced"] = tr["word"]
words=tr["spaced"]
#%%
for i,w in enumerate(words):#range(len(words)):
    let = [l for l in (w)]
    " ".join(let)
    print (i)
    words[i]=let

#%%
txt = tr["word"].str.lower().str.cat()
txt_let = [l for l in (txt)]
" ".join(txt_let)
words_except_stop_dist = nltk.FreqDist(w for w in txt_let) 
uniq_words  = {uw for uw in words_except_stop_dist}
uniq_words

#%%
with open('dictioanry_tr.pickle', 'wb') as f:
    pickle.dump(tr, f)
#with open('deneme.pickle', 'wb') as f:
#    pickle.dump(tw, f)