#%%
def tf_func(WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD,word_letter,size,epc):
#%%  
    import time
    import pandas as pd
    import numpy as np

    start_time = time.process_time()
    
    data = []
    for letter in word_letter:
        for idx, word in enumerate(letter):
            for neighbor in letter[max(idx - WINDOW_SIZE_BACK, 0) : min(idx + WINDOW_SIZE_FORWARD, len(letter)) + 1] : 
                if neighbor != word:
                    data.append([word, neighbor])
    #%%
    df_labeled = pd.DataFrame(data, columns = ['input', 'label'])
    
    #%%
    from sklearn.preprocessing import LabelBinarizer
    x = df_labeled["input"]
    y = df_labeled["label"]
    lb = LabelBinarizer()
    X = lb.fit_transform(x)
    Y = lb.transform(y)
    #%%
    import tensorflow as tf
    from tensorflow import keras
    
    #%%
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=X.shape[1]))
    model.add(keras.layers.Dense(size, activation="relu"))
    model.add(keras.layers.Dense(X.shape[1], activation="softmax"))
    
    optimizer = keras.optimizers.SGD(lr=0.01)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    #%%
    history = model.fit(X, Y, epochs=epc)
    Time_elapsed = time.process_time()-start_time
    print ("\n","="*10,"Time elapsed ==> %0.2f second" % Time_elapsed,"="*10,"\n")
    #%
    model_name = "S" + str(size) + "_B" + str(WINDOW_SIZE_BACK) + "_F" + str(WINDOW_SIZE_FORWARD) + "_E" + str(epc)
    run_id = time.strftime("_%d_%H%M")
    filename = model_name + run_id
    model.save(filename)
    #model = tf.keras.models.load_model("S2_B3_F3_E50_26_2004")
 
    return df_labeled, X,Y,model,Time_elapsed, model_name,lb
#%%
def windowize(word_letter,WINDOW_SIZE_BACK,WINDOW_SIZE_FORWARD ):
    import pandas as pd
    import numpy as np
    
    data = []
    for letter in word_letter:
        for idx, word in enumerate(letter):
            for neighbor in letter[max(idx - WINDOW_SIZE_BACK, 0) : min(idx + WINDOW_SIZE_FORWARD, len(letter)) + 1] : 
                if neighbor != word:
                    data.append([word, neighbor])
    df_labeled = pd.DataFrame(data, columns = ['input', 'label'])
    return df_labeled

#%%
def labelize(df_labeled):
    from sklearn.preprocessing import LabelBinarizer
    x = df_labeled["input"]
    y = df_labeled["label"]
    lb = LabelBinarizer()
    X = lb.fit_transform(x)
    Y = lb.transform(y)
    return lb, X,Y

#%%
def prep_plots(model,lb,model_name):
    #%%
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    
    hidden1 = model.layers[1]
    W1, b1= hidden1.get_weights()
    
    vectors = (W1*1 + b1*1)
    #print(vectors)
    vectors = np.transpose(vectors)
    
    w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
    w2v_df['word'] = lb.classes_
    w2v_df = w2v_df[['word', 'x1', 'x2']]
    #%%
    from sklearn.metrics.pairwise import cosine_similarity
    simM=np.zeros((w2v_df.shape[0],w2v_df.shape[0]))
    for j in range(w2v_df.shape[0]):
        for i in range(w2v_df.shape[0]):
            v1=w2v_df.iloc[j,1:3].values.reshape(1,-1)
            v2=w2v_df.iloc[i,1:3].values.reshape(1,-1)
            sim = cosine_similarity(v1,v2)
            simM[j,i] = sim
    df_sim = pd.DataFrame(simM, index = w2v_df["word"], columns = w2v_df["word"])
    df_sim
    df_sim.to_csv(model_name+".csv")
    #%%
    plt.figure(figsize = (10,7))
    sns.heatmap(df_sim, annot=False)
    plt.savefig(model_name + "_heatmap" + ".png")
    
    df_1=df_sim.loc[["a","o","u","ı"],["e","ö","ü","i"]]
    plt.figure(figsize = (10,7))
    sns.heatmap(df_1, annot=False)
    plt.savefig(model_name + "_sesuyumu" + ".png")
    
    
    #%%
#    for i in range(simM.shape[0]):
#        simM[i,i]=0
    #%%    
    for i in range(df_sim.shape[0]):
        q0=df_sim.columns[i]
        q1=np.argmax(df_sim[q0])
        q2=np.argmax(df_sim[q1])
        q3=np.argmax(df_sim[q2])
        q4=np.argmax(df_sim[q3])
        q5=np.argmax(df_sim[q4])
        q6=np.argmax(df_sim[q5])
        q7=np.argmax(df_sim[q6])
        
        #print(q0,q1,q2,q3,q4,q5,q6,q7)

    #%%
    fig, ax = plt.subplots(figsize = (12,10))
    
    for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
        ax.annotate(word, (x1,x2 ),fontsize=15)
    plt.plot(w2v_df['x1'],w2v_df['x2'],"r.")    
    ax.axhline(y=0, color='b')
    ax.axvline(x=0, color='b')
    #plt.title(model_name) 
    plt.show()
    fig.savefig(model_name+".png")
    
    return df_sim

#%%
def create_w2v(model,lb):
    import pandas as pd
    import numpy as np
    
    hidden1 = model.layers[1]
    W1, b1= hidden1.get_weights()
    vectors = (W1*1 + b1*1)
    #print(vectors)
    vectors = np.transpose(vectors)
    colnames=["a"] * W1.shape[0]
    for i in range(W1.shape[0]):
        colnames[i] = "x"+str(i+1)


    w2v_df2 = pd.DataFrame(lb.classes_,columns = ['word'])
    w2v_df3 = pd.DataFrame(vectors, columns = colnames)
    w2v_df = pd.concat([w2v_df2, w2v_df3], axis=1)

    
#    w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
#    w2v_df['word'] = lb.classes_
#    w2v_df = w2v_df[['word', 'x1', 'x2']]
    return w2v_df
#%%
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

#%% most similar
def show_similar(df_sim,letter,num2show,sort="most"):
    le_sort=df_sim[letter]
    
    if sort=="most": asc=False
    if sort=="least": asc=True
    
    le_sortf=le_sort.sort_values(ascending=asc)
    print(le_sortf[0:num2show])
    return le_sortf