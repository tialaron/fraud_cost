from tensorflow.keras.models import Model, Sequential, load_model # загружаем абстрактный класс базовой модели сети от кераса и последовательную модель
from tensorflow.keras.datasets import mnist, fashion_mnist # загружаем готовые базы mnist
# Из кераса загружаем необходимые слои для нейросети
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D
from tensorflow.keras import backend as K # подтягиваем базовые керасовские функции
from tensorflow.keras.optimizers import Adam # загружаем выбранный оптимизатор
from tensorflow.keras import utils # загружаем утилиты кераса
#from google.colab import files # модуль для загрузки файлов в colab
#import matplotlib.pyplot as plt # из библиотеки для визуализации данных возьмём интерфейс для построения графиков простых функций
from tensorflow.keras.preprocessing import image # модуль для отрисовки изображения
import numpy as np # библиотека для работы с массивами данных
import pandas as pd # библиотека для анализа и обработки данных
#import seaborn as sns
#from PIL import Image # модуль для отрисовки изображения
from sklearn.model_selection import train_test_split # модуль для разбивки выборки на тренировочную/тестовую
from sklearn.preprocessing import StandardScaler # модуль для стандартизации данных
import scipy.stats as stats
import scipy
import streamlit as st
import plotly.figure_factory as ff
import math

from scipy.signal import savgol_filter

df = pd.read_csv("creditcard2.csv") # читаем базу

st.title("Рассмотрим определение мошеннических операций")
'Имеется некий датасет из транзакций.'
data1 = df.drop(['Time'],axis=1)
data1['Amount'] = StandardScaler().fit_transform(data1['Amount'].values.reshape(-1, 1))
st.write(data1)
'Каждая из которых может быть либо нормальной или мошеннической (колонка Class)'

frauds1 = data1[data1.Class == 1]
normal1 = data1[data1.Class == 0]

'Датасет очень неравномерный. Мошеннических операций у нас гораздо меньше чем обычных '
'Мошеннические: ',frauds1.shape
'Нормальные: ',normal1.shape
'Поэтому обучение будет проводиться только на нормальных транзакциях (поиск аномалий)'
percentage1 = st.slider('Разобъем датасет на обучающую и тестовую выборки: ',min_value=5,max_value=95,value=80,step=1)

X_train, X_test = train_test_split(normal1, test_size=(100-percentage1)/100, random_state=42)

X_train = X_train.drop(['Class'], axis=1)
X_test = pd.concat([X_test, frauds1])
y_test = X_test['Class']                    #Проставили метки
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values
'Размер обучающей выборки: ',X_train.shape
'Размер тестовой выборки: ',X_test.shape

model1 = load_model('autocoder1.h5')
prediction1 = model1.predict(X_test)

mse1 = np.mean(np.power(X_test - prediction1, 2), axis=1)
mse_normal = mse1[y_test.values == 0] # среднеквадратичная ошибка на нормальных операциях
mse_frauds = mse1[y_test.values == 1] # среднеквадратичная ошибка на мошеннических операциях

mse_normal = np.log(mse_normal)
mse_frauds = np.log(mse_frauds)
min_mse_norm = min(mse_normal)
max_mse_fraud = max(mse_frauds)

'Минимальная MSE для нормальных операций', min_mse_norm
'Максимальная MSE для мошеннических операций', max_mse_fraud
delta_bias = max_mse_fraud - min_mse_norm

def getAccByBias(bias): # функция будет принимать какое то пороговое значение
  isNormal = mse_normal < bias # если ошибка меньше порога - то транзакция нормальная
  isFrauds = mse_frauds > bias # если ошибка больше порога - то транзакция мошенническая

  accNormal = sum(isNormal) / len(isNormal) # вычисляем процент нормальных операций
  accFaruds = sum(isFrauds) / len(isFrauds) # вычисляем процент мошеннических операций

  'Распознано нормальных транзакций: ', round(100*accNormal), '%'
  'Распознано мошеннических транзакций: ', round(100*accFaruds), '%'
  'Средняя точность распознавания: ', round(50*(accNormal + accFaruds)), '%'

'Также давайте укажем порог чувствительности автокодировщика'
'Все что больше этого порога будет определяться как мошенническая операция'
bias1 = st.slider('Выберите порог: ',min_value=0,max_value=100,value=30,step=1)


bias1_real = (bias1/100)*delta_bias + min_mse_norm
'Получились следующие точности'
getAccByBias(bias1_real)

hist_data = [mse_normal, mse_frauds]
group_labels = ['MSE_Normal', 'MSE_Fraud']
fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1,.1])
fig.add_vline(x=bias1_real)
st.plotly_chart(fig, use_container_width=True)

