
# coding: utf-8

import numpy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error




#将间隔为日的数据转化为周
def day_to_week(data):
    data_week=pd.DataFrame()
    date_list=[]
    quantity_list=[]
    for i in range(0,len(data)):
        s=0
        if i%7==0:
            a=data['日期'][i]
            date_list.append(a)
            quantity_list.append(int(sum(data['数量'][i:i+7])))
    data_week['date']=date_list
    data_week['quantity']=quantity_list
    return(data_week)





# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)



def LSTM_model(data,epochs=100,batch_size=1,verbose=0):
    #data_prepare
    dataset = data['quantity']
    # 将整型变为float
    dataset = dataset.astype('float32')
   # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1,1))


    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:] 
    # use this function to prepare the train and test datasets for modeling
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig("filename.png")
    plt.show()
    
    result_dataframe=[]
    result_dataframe=pd.DataFrame(testPredict[:,0])
    return(result_dataframe)


def main():
    warnings.filterwarnings("ignore")
    #加载数据
    data=pd.read_csv('data_zhuoshini.csv')
    group_dict=dict(data['数量'].groupby(data['品类']).sum())
    group_dict_sorted=sorted(group_dict.items(),key=lambda item:item[1],reverse=True)
    class_list=[]
    #筛选出销量前10名的品类分别保存
    def data_picking(shoes_class):
        locals()[str(shoes_class)]=data[data['品类']==shoes_class][['日期','数量']].sort_values("日期")
        f=str(shoes_class)+'.csv'
        locals()[str(shoes_class)].to_csv(f)
        #return(locals()[str(shoes_class)])
        return(shoes_class)
    for i in range(0,10):
        shoes_class=group_dict_sorted[i][0]
        data_picking(shoes_class)
        class_list.append(shoes_class)
    for each_class in class_list:
        #data_loading
        f=each_class+'.csv'
        data=pd.read_csv(f)
        print(each_class+'Data loading completed!')
        data=day_to_week(data)
        result=LSTM_model(data)
        result.to_csv(each_class+'LSTMresult'+'.csv')

        
    


if __name__ == "__main__":
    look_back=1
    main()

