# Instagram-Fake-Account-Detector

本專題製作一個能幫助判別 IG 假帳號的系統：使用者輸入目標的 IG 帳號 ID，程式將判斷此帳號是真帳號，或是騷擾、色情帳號…等假帳號。

以下為系統網址：https://python-alison.herokuapp.com/


此程式分為三個部分：

1. 機器學習模型
2. 爬蟲獲取資料
3. 使用者介面

# Required Python libraries

* numpy
* pandas
* sklearn
* scipy
* seaborn
* matplotlib
* joblib
* instaloader
* flask
* gunicorn

# 機器學習模型

## Features

選擇不論帳號公開與否皆可取得之公開資訊：

<img src="https://i.imgur.com/SHae8rJ.jpg" width="250" height="150" alt="Features"/><br/>

以下為模型採用的 10 個特徵：

* **Profile Picture**: User has profile picture or not.

* **Num/Length (Username)**: Ratio of number of numerical characters in username to its length.

* **Num/Length (Fullname)**: Ratio of number of numerical characters in fullname to its length.

* **Fullname words**: How many words are in the fullname?

* **Description Length**: How many characters are in the biography?

* **External URL**: User has external URL or not.

* **Private or not**: User is private or not.

* **Number of Posts**: How many posts does the user have?

* **Number of Followers**: How many followers does the user have?

* **Number of Followees**: How many followees does the user have?

## Training

1. Cross Validation → 選出最優模型

   <img src="https://i.imgur.com/GUGUtcY.png" width="180" height="150" alt="Model"/><br/>
   
   RandomForestClassifier 結果最佳，因此後續採用此模型進行訓練。

2. Random Search → 選出最優參數

## result

accuracy: 0.933333

precision: 0.933333

recall: 0.933333

<img src="https://i.imgur.com/DStPPG1.png" width="300" height="250" alt="Result"/><br/>

# 爬蟲獲取資料

Instaloader 爬取模型所需之特徵（欲查詢帳號的資訊），交由模型預測。

# 使用者介面

## 初始畫面

<img src="https://i.imgur.com/iICVKNj.png" width="270" height="150" alt="GUI"/><br/>

## 判斷結果呈現

<img src="https://i.imgur.com/LuGCAhy.jpg" width="700" height="300" alt="output"/><br/>

畫面上顯示預測結果，以及目標帳號的資訊。

# Authors

Ying Hsuan Chen 陳映璇

I Chieh Yang 楊沂潔

# Reference

https://www.kaggle.com/code/khoingo16/predicting-a-fake-instagram-account/notebook
