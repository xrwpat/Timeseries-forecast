# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

warnings.filterwarnings("ignore")

#将间隔为日的数据转化为周
def day_to_week(data):
    #form data from day to week
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
#     data_week.index=data_week.date
    return(data_week)

#计算准确率函数：
def cal_accurate_rate(a,b):
    return(1-abs(a-b)/max(a,b))
def mean_accurate(test,predict):
    try:
        accurate_list=list(map(cal_accurate_rate,test,predict))#list(predict.predicted_mean[-40:]
        accurate_rate=np.mean(accurate_list)
        return(accurate_rate,accurate_list)
        print(accurate_rate)
    except:
        print('The length of test_list and predict_list are not the same. Please check your input.')


#把数据整理成需要的形式
# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 设置随机因子
np.random.seed(7)


def LSTM_model(data, epochs=100, batch_size=1, verbose=0):
    warnings.filterwarnings("ignore")

    # data_prepare
    dataset = data['quantity']
    # 将整型变为float
    dataset = dataset.astype('float32').values
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    # 分割训练集，验证集 和 测试集
    train_size = len(dataset) - 39-54
    val_size=54
    test_size = len(dataset) - train_size
    train, val,test = dataset[0:train_size], dataset[train_size:train_size+val_size], dataset[train_size+val_size:len(dataset), :]
    # use this function to prepare the train and test datasets for modeling
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    valX,valY=create_dataset(val, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    valX= np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(8, input_shape=(look_back,1),recurrent_dropout=0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 绘制训练损失和验证损失
    history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=0,validation_data=[valX,valY])
    history_dict = history.history
    loss_value = history_dict['loss']
    val_loss_value = history_dict['val_loss']

    epochs = range(1, len(loss_value) + 1)
    plt.figure(figsize=(20,20))
    plt.plot(epochs, loss_value, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_value, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # # shift test predictions for plotting
    # testPredictPlot = np.empty_like(dataset)
    # testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    #
    # # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()

    # 生成结果数据
    result_dataframe = []
    accurate_list = list(map(cal_accurate_rate, testPredict[:, 0], testY[0]))
    result_dataframe = pd.DataFrame(testPredict[:, 0], testY[0])
    result_dataframe['accurate_rate'] = accurate_list
    mean_accurate_rate = np.mean(accurate_list)

    # 输出准确率的值
    print('2018年预测准确率为：', mean_accurate_rate)
    return (result_dataframe, trainY[0], trainPredict[:, 0], mean_accurate_rate)


#————————————————————————主程序————————————————————————
def main():
    warnings.filterwarnings("ignore")
    # 加载数据
    data = pd.read_csv('data_zhuoshini.csv')
    group_dict = dict(data['数量'].groupby(data['品类']).sum())
    group_dict_sorted = sorted(group_dict.items(), key=lambda item: item[1], reverse=True)
    class_list = []

    # 筛选出销量前10名的品类分别保存
    def data_picking(shoes_class):
        locals()[str(shoes_class)] = data[data['品类'] == shoes_class][['日期', '数量']].sort_values("日期")
        f = str(shoes_class) + '.csv'
        locals()[str(shoes_class)].to_csv(f)
        # return(locals()[str(shoes_class)])
        return (shoes_class)

    for i in range(0, 10):
        shoes_class = group_dict_sorted[i][0]
        data_picking(shoes_class)
        class_list.append(shoes_class)

    accurate_rate_dict = {}
    s = 0

    for each_class in class_list:
        # data_loading
        f = each_class + '.csv'
        f=open(f,encoding='utf-8')
        data = pd.read_csv(f)
        print(each_class + 'Data loading completed!')
        data = day_to_week(data)
        result, test, predict, mean_accurate_rate = LSTM_model(data, epochs=50, batch_size=1, verbose=0)
        #暂时先不用保存：result.to_csv(each_class + 'LSTMresult2' + '.csv')
        accurate_rate_dict[each_class] = mean_accurate_rate
        s = s + mean_accurate_rate
    accurate_rate_dict['平均准确率'] = s / 10
    print(accurate_rate_dict)

if __name__ == "__main__":
    look_back=1
    main()



