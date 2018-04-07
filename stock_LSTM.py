#importing libraries
import numpy as np
import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#read csv file
djia = pd.read_csv("Combined_News_DJIA.csv")


start1= '2008-08-08'
end1 = '2014-12-31'
start2 = '2015-01-02'
end2 = '2016-07-01'

#create training and testing dataframe on 80 % and 20 % respectively
traindf = djia[(djia['Date']>=start1) & (djia['Date']<=end1)]
testdf = djia[(djia['Date']>=start2) & (djia['Date']<=end2)]

attrib = djia.columns.values

trainX = traindf.loc[:,attrib[2:len(attrib)]]
trainY = traindf.iloc[:,1]

testX = testdf.loc[:,attrib[2:len(attrib)]]
testY = testdf.iloc[:,1]

# =============================================================================
# creating training and testing dataframe on 70 % and 30 % respectively
# start1= '2008-08-08'
# end1 = '2014-03-31'
# start2 = '2014-04-01'
# end2 = '2016-07-01'
# 
# 
# 
# traindf1 = djia[(djia['Date']>=start1) & (djia['Date']<=end1)]
# testdf1 = djia[(djia['Date']>=start2) & (djia['Date']<=end2)]
# 
# attrib = djia.columns.values
# 
# trainX = traindf1.loc[:,attrib[2:len(attrib)]]
# trainY = traindf1.iloc[:,1]
# 
# testX = testdf1.loc[:,attrib[2:len(attrib)]]
# testY = testdf1.iloc[:,1]
# =============================================================================

# =============================================================================
# joining top 25 news to make one sentence
# =============================================================================
def combine(newdf):
    j = newdf.apply(lambda x: ''.join(str(x.values)), axis=1)
    return j
merge_trainX = combine(trainX)
merge_testX = combine(testX)


# =============================================================================
# removing useless symbols from training and testing data
# =============================================================================
def removeuseless(t):
    newt=t.replace('\n','')
    newt=t.replace('"b','')
    newt=t.replace("'b",'')
    for punc in list(punctuation):
        newt=newt.replace(punc,'')
        newt=newt.lower()
    newt=re.sub(' +',' ',newt)
    return newt

merge_trainX = merge_trainX.apply(lambda x: removeuseless(x))
merge_testX = merge_testX.apply(lambda x: removeuseless(x))

# =============================================================================
# removing stopwords from training and testing data
# =============================================================================
temp1=[]
temp2=[]
temptrain=list(merge_trainX)
temptest=list(merge_testX)
s=set(stopwords.words('english'))
for i in temptrain:
    f=i.split(' ')
    for j in f:
        if j in s:
            f.remove(j)
    s1=""
    for k in f:
        s1+=k+" "
    temp1.append(s1)
merge_trainX=temp1

for i in temptest:
    f=i.split(' ')
    for j in f:
        if j in s:
            f.remove(j)
    s1=""
    for k in f:
        s1+=k+" "
    temp2.append(s1)
merge_testX=temp2

# =============================================================================
# tokenizing the training data
# =============================================================================
topwords = 10000
tokenizer = Tokenizer(num_words=topwords)
tokenizer.fit_on_texts(merge_trainX)


# =============================================================================
# generating sequences
# =============================================================================
def seq_gen(text,MAX_SEQUENCE_LENGTH):
     seq= tokenizer.texts_to_sequences(text)
     seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
     return seq
maxwords = 400
seqTrainX = seq_gen(merge_trainX,maxwords)
seqTestX = seq_gen(merge_testX,maxwords)

# =============================================================================
# preparing the model for training and validation
# =============================================================================
seed = 1
np.random.seed(seed)
embedding_dim = 128
model = Sequential()
model.add(Embedding(topwords, embedding_dim, input_length=maxwords))
model.add(LSTM(256,dropout=0.2,recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batchsize = 32

# =============================================================================
# fitting the training and validation data to the model
# =============================================================================
history=model.fit(seqTrainX, trainY, batch_size=batchsize, epochs=6, validation_data=(seqTestX, testY),verbose=1)

# =============================================================================
# calculating the score and accuracy of model
# =============================================================================
score, acc = model.evaluate(seqTestX, testY,
                            batch_size=batchsize)
predict = model.predict_classes(seqTestX, verbose=0)
outputdf=pd.DataFrame({"Date":list(testdf['Date']), "label":list(predict)})
print('Test score:', score)
print('Test accuracy:', acc)

# =============================================================================
# plotting the graph
# =============================================================================

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# =============================================================================
# plot confusion matrix
# =============================================================================
confusion_matrix(testY,predict)

