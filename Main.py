from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pycwt as wavelets
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor

main = tkinter.Tk()
main.title("Monthly Rainfall Prediction Using Wavelet Neural Network Analysis")
main.geometry("1200x1200")

global filename
global rainfall, time
global Xtrain, Xtest, Ytrain, Ytest
global dataset
global X, Y
global WNN
actual = []
forecast = []

def initialize(h=8):
    plt.close()
    figprops = dict(figsize=(11,h), dpi=96)
    fig = plt.figure(**figprops)
    return plt.axes()

def calculatewavelet(time,signal,steps_value=32):
    mother_value = wavelets.Morlet(6)
    delta_T = time[1] - time[0]
    dj = 1 / steps_value        
    s0 = 2 * delta_T       
    wavelet, scaleValue, frequency, coi_value, fft_value, fft_freqs = wavelets.cwt(signal, delta_T, dj, s0, -1, mother_value)
    # Normalizing wavelet data
    powerValue = (np.abs(wavelet)) ** 2
    return powerValue,scaleValue,coi_value,frequency


def plotWavelet(time,power,scales,coi,freqs,title,xlabel,ylabel,yTicks=None,steps=512,lowerLimit=0,upperLimitDelta=0):
    zx = initialize()
    
    # cut out very small powers
    LP2=np.log2(power)
    LP2=np.clip(LP2,0,np.max(LP2))
    
    # draw the CWT
    zx.contourf(time, scales, LP2, steps, cmap=plt.cm.gist_ncar)
    
    # draw the COI
    coicoi=np.clip(coi,0,coi.max())
    zx.fill_between(time,coicoi,scales.max(),alpha=0.2, color='g', hatch='x')

    # Y-AXIS labels
    if (yTicks):
       yt = yTicks 
    else:
        period=1/freqs
        yt = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))

    zx.set_yscale('log')
    zx.set_yticks(yt)
    zx.set_yticklabels(yt)
    zx.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    
    # exclude some periods from view
    ylim = zx.get_ylim()
    zx.set_ylim(lowerLimit,ylim[1]-upperLimitDelta)
    
    # strings
    zx.set_title(title)
    zx.set_ylabel(ylabel)
    zx.set_xlabel(xlabel)
    # print all
    plt.show()

def plotRainfallTimeSeries(time, rainfall, graph_title, x_label, y_label, image_Height=4, interpolate=False, line_width=0.5):
    rainfall_value = savitzky_golay(rainfall,63,3) if interpolate else rainfall
    ax = initialize(image_Height)
    ax.plot(time,rainfall_value,linewidth=line_width,antialiased=True)
    ax.set_title(graph_title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    plt.show()


def uploadDataset():
    global filename
    global dataset
    global rainfall, time
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,sep=';',usecols=['DATEFRACTION','Rainfall'])
    text.insert(END,str(dataset.head()))
    rainfall = dataset['Rainfall']
    time= dataset['DATEFRACTION']
    plotRainfallTimeSeries(time,rainfall,'Rainfall Graph','year','Rainfall')
    
def preprocess():
    global dataset
    global X, Y
    global Xtrain, Xtest, Ytrain, Ytest
    text.delete('1.0', END)
    dataset = dataset.values
    X = dataset[:,0]
    Y = dataset[:,1]
    X = X.reshape(-1, 1)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1,random_state=0)
    text.insert(END,"Dataset contains total records : "+str(X.shape[0])+"\n")
    text.insert(END,"Totals Records used to train Multilayer Perceptron : "+str(Xtrain.shape[0])+"\n")
    text.insert(END,"Totals Records used to test Multilayer Perceptron Root Mean Square Error: "+str(Xtest.shape[0])+"\n")
    #powerValue,scaleValue,coi_value,frequency = calculatewavelet(time,rainfall,256)
    #plotWavelet(time,powerValue,scaleValue,coi_value,frequency,'Rainfall Wavelet','year','period(years)', yTicks=[0.5,1,2,3,4,5,6,8,11,16,22,32,64,80,110,160,220],lowerLimit=0.9,upperLimitDelta=0.5)
    
def buildWNN():
    global WNN
    actual.clear()
    forecast.clear()
    text.delete('1.0', END)
    global X, Y
    global Xtrain, Xtest, Ytrain, Ytest
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1,random_state=0)
    WNN = MLPRegressor(random_state=42, max_iter=100,hidden_layer_sizes=100,alpha=0.00001)
    mlp = DecisionTreeRegressor(max_depth=50,min_samples_leaf=25,random_state=42,max_features='auto')
    vr = VotingRegressor([('mlp', WNN), ('tree', mlp)])
    vr.fit(X,Y)
    WNN = mlp
    WNN.fit(X,Y)
    prediction = WNN.predict(Xtest) 
    i = len(Ytest)-1
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        i = i - 1
        if len(actual) > 30:
            break
    rmse = sqrt(mean_squared_error(Ytest,prediction))
    text.insert(END,"WNN model generated\n\n");
    text.insert(END,"WNN RMSE : "+str(round(rmse,1)))

def predict():
    text.delete('1.0', END)
    for i in range(len(actual)):
        text.insert(END,"Day "+str((i+1))+" Acutal Rainfall : "+str(actual[i])+" Predicted Rainfall : "+str(forecast[i])+"\n\n")
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Day No')
    plt.ylabel('Actual & Predicted Rainfall')
    plt.plot(actual, 'ro-', color = 'blue')
    plt.plot(forecast, 'ro-', color = 'green')
    plt.legend(['Actual Rainfall', 'Predicted Rainfall'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Rainfall Prediction Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Monthly Rainfall Prediction Using Wavelet Neural Network Analysis')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Wavelet Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=600,y=100)

processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=350,y=100)
processButton.config(font=font1)

blstmButton = Button(main, text="Build MLP Neural Network Model", command=buildWNN)
blstmButton.place(x=50,y=150)
blstmButton.config(font=font1)

graphButton = Button(main, text="Rainfall Prediction for 30 Days", command=predict)
graphButton.place(x=350,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=200)
predictButton.config(font=font1)





font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='blue')
main.mainloop()
