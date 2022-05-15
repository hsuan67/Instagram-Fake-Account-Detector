# Instagram-Fake-Account-Detector

本專題製作一個能幫助判別IG假帳號的系統：使用者輸入目標的IG帳號ID，程式將判斷此帳號是真帳號，或是騷擾、色情帳號…等假帳號。

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

![image](https://user-images.githubusercontent.com/86561823/168461585-6e59a3f1-2e66-410b-b53d-e5a14f870ee0.png =500x)

選擇不論帳號公開與否皆可取得之公開資訊。

* Username
* Fullname
* Biography word numbers
* Post numbers
* Follower numbers
* Followee numbers
* Have profile picture or not
* Have URL or not
* Private or not

## Training

1. Cross Validation
2. Random Search

## result

# 爬蟲獲取資料
instaloader 抓取目標帳號的資料，作為系統判斷的feature

## 使用者介面
### 初始畫面

### 判斷結果呈現

# Authors
Ying Hsuan Chen 陳映璇

I Chieh Yang 楊沂潔

## References
