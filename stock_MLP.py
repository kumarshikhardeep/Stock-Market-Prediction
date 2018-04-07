import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =============================================================================
# read csv file
# =============================================================================
djia = pd.read_csv("Combined_News_DJIA.csv")

start1= '2008-08-08'
end1 = '2014-12-31'
start2 = '2015-01-02'
end2 = '2016-07-01'

# =============================================================================
# create training and testing dataframe on 80 % and 20 % respectively
# =============================================================================
traindf = djia[(djia['Date']>=start1) & (djia['Date']<=end1)]
testdf = djia[(djia['Date']>=start2) & (djia['Date']<=end2)]

attrib = djia.columns.values

trainX = traindf.loc[:,attrib[2:len(attrib)]]
trainY = traindf.iloc[:,1]

testX = testdf.loc[:,attrib[2:len(attrib)]]
testY = testdf.iloc[:,1]

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

tfidfTrainX=tokenizer.texts_to_matrix(merge_trainX,mode='tfidf')
tfidfTestX=tokenizer.texts_to_matrix(merge_testX,mode='tfidf')

# =============================================================================
# creating catagorical data
# =============================================================================
nb_classes=2
train_Y = np_utils.to_categorical(trainY, nb_classes)
test_Y = np_utils.to_categorical(testY, nb_classes)

# =============================================================================
# preparing the model for training and validation
# =============================================================================
model = Sequential()
model.add(Dense(512, input_dim=tfidfTrainX.shape[1],activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'] )

# =============================================================================
# fitting the training and validation data to the model
# =============================================================================
history2=model.fit(tfidfTrainX, train_Y, epochs=9, batch_size=100,validation_data=(tfidfTestX, test_Y))

#plotting the graph
#summarize history for accuracy
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('MLP model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('MLP model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
