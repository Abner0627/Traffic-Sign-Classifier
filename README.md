# 自動駕駛實務 交通號誌辨識<br />Traffic-Sign-Classifier

## Github

[<img src=https://i.imgur.com/3aZfqpy.png width=15%>](https://github.com/Abner0627/Traffic-Sign-Classifier)

## 作業目標

辨識總計43類之交通號誌標示。

## 環境設定與套件安裝

1. 使用環境：  
    * Win 10 
    * python 3.8.5

2. 進入該專案之資料夾
`cd /d [path/to/this/project]`

3. 安裝所需套件 **(不含pytorch)**
`pip install -r requirements.txt`

4. 至官網安裝Pytorch
依官方下載頁面 ([Link](https://pytorch.org/get-started/locally/))，選擇與系統環境相符的版本下載即可，\
以Win 10且具GPU (CUDA 11.1)為例：\
`pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

5. (選用) 安裝CUDA等所需套件
若要**使用tensorflow並調用GPU**時，則須依官網 ([Link](https://www.tensorflow.org/install/gpu?hl=zh-tw#software_requirements))指示安裝對應套件。

### 資料集分析



### 程式碼註解