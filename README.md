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

選擇不論帳號公開與否皆可取得之公開資訊：

<img src="https://i.imgur.com/SHae8rJ.jpg" width="200" height="100" alt="Features"/><br/>

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

<img src="https://i.imgur.com/GUGUtcY.png" width="150" height="200" alt="Model"/><br/>

2. Random Search → 選出最優參數

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
