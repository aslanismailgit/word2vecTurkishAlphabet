import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import pickle
import time
from harf_gen_func import *
#word2vec_model.init_sims(replace=True)
#https://www.kernix.com/article/similarity-measure-of-textual-documents/

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
#%%
import matplotlib

matplotlib.style.use('ggplot')
rslt.plot.bar(rot=0)
plt.xlabel("letters")
plt.title ("Frequency of Letters")
plt.savefig("Frequency Plot.png")
plt.show()

#%%
word_letter = tr_df["spaced"]

#%% BUILD MODEL USING GENSIM 
import gensim
start_t=time.process_time()

size=[2,2,2,2,100,100,100,100];
wind=[1,2,3,4,1,2,3,4]
min_count=0
itr=100

for i in range(len(size)):
    print("---",size[i],"---",wind[i],"---")
    model_name = "./gensim_models/GEN_S" + str(size[i]) + "_BF" + str(wind[i]) + "_E" + str(itr)
    run_id = time.strftime("_%d_%H%M")
    filename = model_name + run_id
    model = gensim_model(word_letter,size[i], wind[i], min_count,itr)
    model.save(filename)
    #model = Word2Vec.load(filename)
    w2v_df = create_w2v(model)
    w2v_df.to_csv(filename + "w2v.csv")
    df_sim = create_simm(w2v_df)
    df_sim.to_csv(filename+"sim.csv")
    Time_elaps = time.process_time()-start_t
    print ("\n","="*10,"Time elapsed this run ==> %0.2f second" % Time_elaps,"="*10,"\n")
    
Time_elaps = time.process_time()-start_t
print ("\n","="*10,"Total time elapsed ==> %0.2f second" % Time_elaps,"="*10,"\n")

#%%
import os
files = os.listdir('./gensim_models/')
sims=[]
w2vs=[]
for file in files:
    if "csv" in file:
        if "sim" in file:
            sims.append(file)
        if "w2v" in file:
            w2vs.append(file)
        print (file)
#%%
for sim in sims:
    for w in w2vs:
        sn=sim.split("_")
        wn=w.split("_")
        if (sn[1]==wn[1])&(sn[2]==wn[2]):
            df_sim=pd.read_csv("./gensim_models/" + sim)
            df_sim=df_sim.set_index('word')            
            w2v_df=pd.read_csv("./gensim_models/" + w)
            model_name=sn[0]+"_"+sn[1]+"_"+sn[2]+"_"+sn[3]
            prep_plots(df_sim,w2v_df,model_name)
            print(sim)
            print(w, model_name)
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("./gensim_models/GEN_S2_BF2_E100_29_2127sim.csv")
df_sim=df.set_index('word')            

plt.figure(figsize = (10,7))
sns.heatmap(df_sim, annot=False)
plt.ylabel("letters")
plt.xlabel("letters")
plt.title ("Similarties Between Letters")
#%%

sim ve w2v de index problemi var