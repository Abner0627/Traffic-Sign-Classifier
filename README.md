# 自動駕駛實務 交通號誌辨識<br />Traffic-Sign-Classifier

## Github

[<img src=https://i.imgur.com/3aZfqpy.png width=15%>](https://github.com/Abner0627/Traffic-Sign-Classifier)

## 作業目標

辨識總計43類之交通號誌標示。

## 環境設定與套件安裝

**1. 使用環境**
Win 10 
python 3.8.5

**2. 進入該專案之資料夾**
`cd /d [path/to/this/project]`

**3. 安裝所需套件 (不含pytorch)**
`pip install -r requirements.txt`

**4. 至官網安裝Pytorch**
依官方下載頁面 ([Link](https://pytorch.org/get-started/locally/))，選擇與系統環境相符的版本下載即可，\
以Win 10且具GPU (CUDA 11.1)為例：\
`pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

**5. (選用) 安裝CUDA等所需套件**
若要**使用tensorflow並調用GPU**時，則須依官網 ([Link](https://www.tensorflow.org/install/gpu?hl=zh-tw#software_requirements))指示安裝對應套件。

### 資料集分析

**1. 各類別可視化**
    本資料集共蒐集43類不同的德國交通號誌。\
    從各類別隨機取樣40筆作圖可得下圖：
    <img src=https://i.imgur.com/7uPPoyC.png>
    由此可知，收集而成的資料有不同時間段或者光影環境所組成，\
    因此能提升模型的強健性。\
    \
    如果要再進一步強化模型的學習能力，應再對資料作增強 (data augmentation)，\
    如對圖片作旋轉任意角度或水平與垂直翻轉。

**2. 各類別數量統計**
    統計訓練集與驗證集各類別的數量，化成直方圖如下：
    <img src=https://i.imgur.com/vRVy8tG.png>
    <img src=https://i.imgur.com/xjjkOtl.png>
    由此發現各類別的分佈相當不平均，若要增加分類的準確率，\
    應對各類別有所增減使彼此數量相近，避免模型學習時過於偏向某一類別。

3. 圖片數值分佈
   隨機取樣三張圖片，統計其內的數值分佈可畫成以下直方圖：
   <img src=https://i.imgur.com/KN6rK6Q.png>

### 程式碼註解