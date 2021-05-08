# 自動駕駛實務 交通號誌辨識<br />Traffic-Sign-Classifier

## Github Repository

[<img src=https://i.imgur.com/3aZfqpy.png width=25%>](https://github.com/Abner0627/Traffic-Sign-Classifier)

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

**3. 圖片數值分佈**
隨機取樣三張圖片，統計其內的數值分佈可畫成以下直方圖：

<img src=https://i.imgur.com/UsUmcxk.png>

為了增加訓練時的收斂速度與平穩度，故須對各張圖片進行特徵縮放 (feature scaling)，\
此處採用Z-Score Standardization進行，\
將資料轉換為常態分佈，可視化後如下：

<img src=https://i.imgur.com/fIFDe2X.png>

如此便能開始進行模型的訓練。

### 模型介紹

本次作業使用2015年於提出的深度殘差網路，ResNet$^{[1]}$。\
為求節省訓練時間及減少overfitting的影響，\
此處精簡了原網路架構改以13層進行訓練。\
\
於原論文中提到，加深後的神經網路有時表現反而會不如淺層網路，\
也就是說隨著架構變深，擷取出的特徵不一定會變好，\
故作者採用殘差連接的方式，在擷取特徵的同時也能保留輸入的原特徵。\
\
此處選用的ResNet是由多個ResNet block組成，主要有下列兩類：

<img src=https://i.imgur.com/QQtKxK0.png>

與左側相比，右側在進行殘差連接的時候，\
另使用1x1的convolution進行擷取。\
完整的模型架構如下，主要是基於ResNet-12進行修改，\
為使RGB與灰階圖的特徵分開擷取，之後再將特徵串接，\
故再增加一層convolution，總計有13層。

<img src=https://i.imgur.com/nCA9B3A.png>

此外這次作業也分別利用pytorch與keras實現上述模型架構，\
詳見`model_pt.py`與`model_tf.py`。

### 訓練細節

本次作業首先對各張圖片的每個channel進行特徵縮放，\
RGB有3個channel而灰階圖片則有1個channel。\
\
為使模型快速收斂，並減少調整參數的麻煩程度，\
故選用Adam當作optimizer。\
然而隨著訓練時間增加gradient也會縮小，\
但此時Adam反而會受到小的gradient影響，使測試時的表現不佳；\
因此這裡採用2018年提出的改良版本，AMSGrad$^{[2]}$，以減緩該問題。\
\
各訓練參數如下表所示：
| Batch size | Learning rate | Epoch |
|------------|---------------|-------|
| 32         | 0.001         | 20    |

另外訓練時每個epoch的loss與跟驗證集的準確度，＼
畫成折線圖後如下，以使用的框架分成`model_pt`與`model_tf`：

<img src=https://i.imgur.com/YULaote.png>
<img src=https://i.imgur.com/asYvhWH.png>


### 程式碼註解