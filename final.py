from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd
import os 
import librosa 
import wave
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import rmsprop
import IPython.display as ipd

main = tkinter.Tk()
main.title("Speech Recognition")
main.geometry("1300x1200")

def lstmmodel():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    
    # Configures the model for training
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.load_weights("Model_LSTM.h5")
    return model

def upload():
    global filename
    print("Testing")
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"File Seletced loaded\n\n")

def loadmodel():
    
    global model
    model=lstmmodel()
    text.insert(END,"LSTM model Loaded\n\n")


def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name,duration=3
                                  ,offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    return mfccs

def preprocess():
    global filename
    global qq
    a = extract_mfcc(filename)
    a1 = np.asarray(a)

    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    text.insert(END,"Audio Features extracted\n\n")

def pred():
    global model,qq
    pred = model.predict(qq)
    preds=pred.argmax(axis=1)
    classess=['neutral', 'calm', 'happy','sad', 'angry', 'fearful', 'disgust', 'surprised']
    text.insert(END,"Speech predicted :"+str(classess[preds[0]])+"\n")


font = ('times', 16, 'bold')
title = Label(main, text='Clustering-Based Speech Emotion Recognition by Incorporating Learned Features and Dee')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
lm = Button(main, text="Model load", command=loadmodel)
lm.place(x=700,y=100)
lm.config(font=font1)

ml = Button(main, text="Upload Image", command=upload)
ml.place(x=700,y=150)
ml.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=200)

imp= Button(main, text="Audio Preprocess", command=preprocess)
imp.place(x=700,y=250)
imp.config(font=font1)

pt = Button(main, text="Speech Recognition", command=pred)
pt.place(x=700,y=300)
pt.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='bisque2')
main.mainloop()




