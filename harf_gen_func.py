#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#import seaborn as sns
#import re
#from nltk.tokenize import word_tokenize
#import nltk
#import pickle
#import time

#%%--------------------------------------------------
def create_simm(w2v_df):
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    simM=np.zeros((w2v_df.shape[0],w2v_df.shape[0]))
    
    for j in range(w2v_df.shape[0]):
        for i in range(w2v_df.shape[0]):
            v1=w2v_df.iloc[j,1:3].values.reshape(1,-1)
            v2=w2v_df.iloc[i,1:3].values.reshape(1,-1)
            sim = cosine_similarity(v1,v2)
            simM[j,i] = sim
    df_sim = pd.DataFrame(simM, index = w2v_df["word"], columns = w2v_df["word"])
    return df_sim
##%--------------------------------------------
#df_sim2=pd.DataFrame(index=words, columns=words)
#for i,w1 in enumerate(words):
#    for j,w2 in enumerate(words):
#      m=model.wv.similarity(w1,w2)
#      df_sim2.iloc[i,j]=m
#      print(m)
#df_sim2
#%%---------------------------------------------
def create_w2v(model):
    import pandas as pd
    words = list(model.wv.vocab)
    X = model.wv[model.wv.vocab]
    vectors = (X)
    colnames=["a"] * X.shape[1]
    for i in range(X.shape[1]):
        colnames[i] = "x"+str(i+1)
    w2v_df2 = pd.DataFrame(words,columns = ['word'])
    w2v_df3 = pd.DataFrame(vectors, columns = colnames)
    w2v_df = pd.concat([w2v_df2, w2v_df3], axis=1)
    return w2v_df
#%%----------------------------------------------
def gensim_model(word_letter,size, wind,min_count,itr):
    import gensim
    model = gensim.models.Word2Vec(word_letter,
                                   size=size,
                                   window=wind,
                                   min_count=min_count,
                                   workers=1, iter=itr)

    return model
#%%------------------------------------------------
def prep_plots(df_sim,w2v_df,model_name):
    #%%
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
#    df_sim=df_sim.set_index('word')
    
##%% ----- heat map -----------
    plt.figure(figsize = (10,7))
    sns.heatmap(df_sim, annot=False)
    plt.ylabel("letters")
    plt.xlabel("letters")
    plt.title ("Similarties Between Letters")
    plt.savefig(model_name + "_heatmap" + ".png")
    
    df_1=df_sim.loc[["a","o","u","ı"],["e","ö","ü","i"]]
    plt.figure(figsize = (10,7))
    sns.heatmap(df_1, annot=False)
    plt.ylabel("letters")
    plt.xlabel("letters")
    plt.title ("Similarties Between Vowels")
    plt.savefig(model_name + "_sesuyumu" + ".png")
    print("Heatmap saved.......")

#%% --- 
    fig, ax = plt.subplots(figsize = (12,10))
    
    for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
        ax.annotate(word, (x1,x2 ),fontsize=15)
    plt.plot(w2v_df['x1'],w2v_df['x2'],"r.")    
    ax.axhline(y=0, color='b')
    ax.axvline(x=0, color='b')
    plt.ylabel("")
    plt.xlabel("")
    plt.title ("")
    plt.show()
    fig.savefig(model_name+".png")
    print("Similarity plot saved.......")

