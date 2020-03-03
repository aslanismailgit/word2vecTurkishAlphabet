import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#import re
from nltk.tokenize import word_tokenize
import nltk
import pickle
import time
from harf_tf_func import *
#%%
with open('./data/dictioanry_tr.pickle', 'rb') as f:
    tr_df = pickle.load(f)

#%%
txt = tr_df["word"].str.lower().str.cat()
txt_let = [l for l in (txt)]
" ".join(txt_let)
words_dist = nltk.FreqDist(w for w in txt_let) 
uniq_words  = {uw for uw in words_dist}

rslt = pd.DataFrame(words_dist.most_common(len(uniq_words)),
                    columns=['Word', 'Frequency']).set_index('Word')
matplotlib.style.use('ggplot')
rslt.plot.bar(rot=0)
plt.savefig("Frequency Plot.png")


#%
word_letter = tr_df["spaced"]
#%%
start_t = time.process_time()

epc = 50
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 1
WINDOW_SIZE_FORWARD = 1
size = 2
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 2
WINDOW_SIZE_FORWARD = 2
size = 2
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 3
WINDOW_SIZE_FORWARD = 3
size = 2
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 3
WINDOW_SIZE_FORWARD = 0
size = 2
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 1
WINDOW_SIZE_FORWARD = 0
size = 2
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 1
WINDOW_SIZE_FORWARD = 1
size = 100
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
#df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 2
WINDOW_SIZE_FORWARD = 2
size = 100
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
#df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 3
WINDOW_SIZE_FORWARD = 3
size = 100
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
#df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 3
WINDOW_SIZE_FORWARD = 0
size = 100
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
#df_sim = prep_plots(model,lb,model_name)
#--------------------------------------------------------------------
WINDOW_SIZE_BACK = 1
WINDOW_SIZE_FORWARD = 0
size = 100
df_labeled, X,Y,model,Time_elapsed, model_name,lb = tf_func(
        WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc)
 
#df_sim = prep_plots(model,lb,model_name)
Time_elaps = time.process_time()-start_t
print ("\n","="*10,"Time elapsed ==> %0.2f second" % Time_elaps,"="*10,"\n")

#%%
letter="a"
num2show=10
sort = "most"
#sort =  "least"
le_sortf = show_similar(df_sim,letter,num2show,sort)    
plt.plot(le_sortf,".")
plt.savefig(model_name + "__a similarity.png")

#model = tf.keras.models.load_model("S2_B3_F3_E50_26_2004")
