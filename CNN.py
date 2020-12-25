import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import cv2
import os



(x_Train,y_Train),(x_Test,y_Test)=keras.datasets.mnist.load_data()
# X=sample  Y=label
print('x_Train_img',x_Train.shape)
print('y_Train_img',y_Train.shape)

print('x_Test_img',x_Test.shape)
print('y_Test_img',y_Test.shape)

# 將RGB3維轉成一維
print(x_Train.shape)
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
print('x_Train4D.shape ',x_Train4D.shape)
print('x_Test4D.shape ',x_Test4D.shape)

# 將資料正規畫到0~1之間
x_Train4D_nor=x_Train4D/255
x_Test4D_nor=x_Test4D/255

# 重新編碼ONEHOT
y_Train_onehot=np_utils.to_categorical(y_Train)
y_Test_onehot=np_utils.to_categorical(y_Test)
print(y_Train_onehot.shape)


# # CNN model
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_nor,y=y_Train_onehot,validation_split=0.2,epochs=2,batch_size=300,verbose=2)
predictions = model.predict_classes(x_Test4D_nor)
print(len(predictions))
pd.crosstab(y_Test,predictions,rownames=['實際值'],colnames=['預測值'])
