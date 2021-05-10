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

<img src=https://i.imgur.com/NAR61eY.png>
<img src=https://i.imgur.com/IMwujpb.png>

### 模型驗證
執行`test_01.py`，並選擇使用pytorch (`-P True`)或keras (`-T True`)進行驗證\
(需先執行對應的訓練程式，`main_pt.py`或`main_tf.py`)；\
以pytorch來說，需在terminal輸入`python test_01.py -P True`即可。\
\
兩者在`test.p`的預測準確度如下：
Pytorch

<img src=https://i.imgur.com/JjPAvsj.png>

Keras

<img src=https://i.imgur.com/KsaZLft.png>

\
另從網路上隨機擷取5張道路標誌圖進行驗證，即`test_02.py`，\
操作方式同`test_01.py`，進行選擇pytorch或keras。\
驗證完後會輸出對應的預測結果，並以`.npy`形式存於`./test_img/result`，\
之後直接執行`python test_02.py`出圖，並以藍色與紅色長條圖區分預測正確或錯誤。
Pytorch

<img src=https://i.imgur.com/bS97epM.png>

Keras

<img src=https://i.imgur.com/dsTCrul.png>

### 程式碼註解
分作pytorch以及keras進行說明，各參數詳見`config.py`。

**1. Pytorch**
**(1) 讀取檔案與前處理**
```py
# 檔案路徑與檔名
dpath = './traffic-signs-data'
traF = 'train.p'
valF = 'valid.p'

with open(os.path.join(dpath, traF), 'rb') as f:
    traD = pickle.load(f)
with open(os.path.join(dpath, valF), 'rb') as f:
    valD = pickle.load(f)
# RGB圖片
traRGB = traD['features'].transpose(0,3,1,2)
valRGB = valD['features'].transpose(0,3,1,2)
# 轉成灰階圖
traGray = func._gray(traRGB)
valGray = func._gray(valRGB)
# 將RGB與灰階圖串接
traImg = np.concatenate((traRGB, traGray), axis=1)
valImg = np.concatenate((valRGB, valGray), axis=1)
# 對每張圖片正規化後轉成tensor
tra_data = torch.from_numpy(func._norm(traImg))
           .type(torch.FloatTensor)
val_data = torch.from_numpy(func._norm(valImg))
           .type(torch.FloatTensor)
# 標籤轉成tensor
tra_label = torch.from_numpy(traD['labels'])
           .type(torch.FloatTensor)
val_label = torch.from_numpy(valD['labels'])
           .type(torch.FloatTensor)
# 包成pytorch的dataset形式
tra_dataset = torch.utils.data.TensorDataset(tra_data, tra_label)
val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
# 包成pytorch的dataloader形式並設定batch size，以進行訓練
tra_dataloader = torch.utils.data.DataLoader(dataset=tra_dataset,
                 batch_size=config.batch, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                 batch_size=config.batch, shuffle=False)
```
此步驟會將圖片之RGB與灰階圖串接在一起後，\
為加快模型收斂，故再對圖片進行正規化。\
此外pytorch提供了`dataset`和`dataloader`形式方便使用者訓練，\
於該處可設定batch size以及是否進行shuffle。

**(2) 設定參數**
```py
# 讀取模型、設定optimizer與loss function
model = model_pt.ResNet18()
optim_m = optim.Adam(model.parameters(), lr=config.lr,
          amsgrad=config.amsgrad)
loss_func = nn.CrossEntropyLoss()
# 將模型與loss function放進GPU進行計算
# (若無GPU則會自動使用CPU)
model = model.to(device)
loss_func = loss_func.to(device)
```

**(3) 模型訓練**
```py
#%% Training
L, A = [], []
for epoch in range(config.Epoch):
    # 將模型切換為訓練模式
    model.train()
    for ntra, (Data, Label) in enumerate(tra_dataloader):
        # 初始化optimizer
        optim_m.zero_grad()
        # 將資料與標籤放入GPU進行計算
        data_rgb = Data[:,:3,:,:].to(device)
        data_gray = Data[:,-1,:,:].unsqueeze(1)
        data_gray = data_gray.to(device)
        val = Label.type(torch.long).to(device)
        pred = model(data_rgb, data_gray)
        # 計算loss
        loss = loss_func(pred, val)
        # Backpropagation
        loss.backward()
        optim_m.step()
```
與keras相比，pytorch訓練過程較為詳細。\
首先會先從初始化optimizer開始，再由模型輸出預測結果，\
接著計算loss及進行Backpropagation。\
\
另外在模型架構方面，\
pytorch是以`(batch size, channel, H, W)`的方式排列；\
此點與keras`(batch size, H, W, channel)`的形式不同。


**(4) 模型驗證**
```py
# 將模型切換為評估模式
model.eval()
with torch.no_grad():
    for nval, (Data_V, Label_V) in enumerate(val_dataloader):
        data_rgb = Data_V[:,:3,:,:].to(device)
        data_gray = Data_V[:,-1,:,:].unsqueeze(1)
        data_gray = data_gray.to(device)
        pred = model(data_rgb, data_gray)
        # 將預測結果放回CPU並轉成numpy
        out = pred.cpu().data.numpy()
        pr  = np.argmax(out, axis=1)
        if nval==0:
            prd = pr
        else:
            prd = np.concatenate((prd, pr))
    # 計算預測與驗證集的準確度，
    # 並print出每個epoch的loss及準確度
    va = valD['labels']
    hd = np.sum(prd==va)
    acc = (hd/va.shape[0])
    loss_np = loss.item()
    print('epoch[{}] >> loss:{:.4f}, val_acc:{:.4f}'
            .format(epoch+1, loss_np, acc)) 

    L.append(loss_np)
    A.append(acc)
```
另一方面，為提供除loss外評估模型的方式，\
故在每個epoch訓練完後，再切換為驗證模式與驗證集計算準確度。\

**(5) 測試模型 (test_01.py)**
```py
dpath = './traffic-signs-data'
tesF = 'test.p'
M = './model'

with open(os.path.join(dpath, tesF), 'rb') as f:
    tesD = pickle.load(f)

# Img
tesRGB = tesD['features'].transpose(0,3,1,2)
tesGray = func._gray(tesRGB)

tes_data = func._norm(np.concatenate((tesRGB, tesGray, 
           axis=1))
tes_label = tesD['labels']
```
同樣讀取資料並進行前處理。

```py
tesDa = torch.from_numpy(tes_data).type(torch.FloatTensor)
tesLa = torch.from_numpy(tes_label).type(torch.FloatTensor)
tes_dataset = torch.utils.data.TensorDataset(tesDa, tesLa)
tes_dataloader = torch.utils.data.DataLoader(dataset=tes_dataset, 
                 batch_size=32, shuffle=False)
# 讀取訓練好的模型
model = torch.load(os.path.join(M, 'model_pt.pth'))
model.cpu()
model.eval()
with torch.no_grad():
    for ntes, (Data_E, Label_E) in enumerate(tes_dataloader):
        data_rgb = Data_E[:,:3,:,:]
        data_gray = Data_E[:,-1,:,:].unsqueeze(1)
        pred = model(data_rgb, data_gray)

        out = pred.cpu().data.numpy()
        pr  = np.argmax(out, axis=1)
        if ntes==0:
            prd = pr
        else:
            prd = np.concatenate((prd, pr))

    te = tesD['labels']
    hd = np.sum(prd==te)
    acc = (hd/te.shape[0])

    print('\n=========================')
    print('test_acc >> {:.4f}'.format(acc)) 
    print('=========================')
```
基本上與驗證時大同小異。\
\
**2. Keras**
**(1) 讀取檔案與前處理**
```py
#%% Load
dpath = './traffic-signs-data'
traF = 'train.p'
valF = 'valid.p'

with open(os.path.join(dpath, traF), 'rb') as f:
    traD = pickle.load(f)
with open(os.path.join(dpath, valF), 'rb') as f:
    valD = pickle.load(f)

# Img
traRGB = traD['features'].transpose(0,3,1,2)
valRGB = valD['features'].transpose(0,3,1,2)

traGray = func._gray(traRGB)
valGray = func._gray(valRGB)

tra_data = func._norm(np.concatenate((traRGB, traGray),
           axis=1)).transpose(0,2,3,1)
val_data = func._norm(np.concatenate((valRGB, valGray),
           axis=1)).transpose(0,2,3,1)
tra_label = traD['labels']
val_label = valD['labels']
# 將資料轉成tf.float16進行訓練
tra_data = tf.image.convert_image_dtype(tra_data,
           dtype=tf.float16, saturate=False)
val_data = tf.image.convert_image_dtype(val_data,
           dtype=tf.float16, saturate=False)
```

**(2) 設定參數**
```py
bz = config.batch
model = model_tf.ResNet18()
optim_m = keras.optimizers.Adam(learning_rate=config.lr,
          amsgrad=config.amsgrad)
```

**(3) 模型訓練與驗證**
```py
# 載入optimizer與loss function
model.compile(optimizer=optim_m, 
              loss=keras.losses.SparseCategoricalCrossentropy(
                 from_logits=True), 
              metrics=['accuracy'])
# 訓練模型並記錄各epoch的loss與accuracy
history = model.fit(tra_data, tra_label, batch_size=bz,
                    epochs=config.Epoch, verbose=2, shuffle=True,
                    validation_data=(val_data, val_label))
# 以numpy array形式記錄，方便儲存。
loss = np.array(history.history['loss'])
val_acc = np.array(history.history['val_accuracy'])
```

**(4) 測試模型 (test_01.py)**
```py
tesDa = tes_data.transpose(0,2,3,1)
tesDa = tf.image.convert_image_dtype(tesDa, dtype=tf.float16, saturate=False)
model = keras.models.load_model(os.path.join(M, 'model_tf'))
pro_model = keras.Sequential([model, keras.layers.Softmax()])
pred = pro_model.predict(tesDa)
pro_pred = np.argmax(pred, axis=1)

hd = np.sum(pro_pred==tes_label)
acc = (hd/tes_label.shape[0])

print('\n=========================')
print('test_acc >> {:.4f}'.format(acc)) 
print('=========================')
```
讀取資料部分與pytorch一樣
