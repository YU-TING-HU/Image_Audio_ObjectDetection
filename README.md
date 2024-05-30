跳至 [Object Detection using YOLOv7 and YOLOv8](#object-detection-using-yolov7-and-yolov8)

跳至 [Audio Classification with CNN](#audio-classification-with-cnn)

---

# Object Detection using YOLOv7 and YOLOv8

使用 YOLOv7 和 YOLOv8 模型偵測圖片上的人是否戴口罩。

## 目錄

- [套件](#套件)
- [分析流程](#分析流程)

## 套件

- Pandas
- NumPy
- Seaborn
- ultralytics（用於 YOLOv8）

```bash
pip install pandas numpy seaborn ultralytics
```

## 分析流程

- [下載預訓練模型和建立資料夾目錄結構](#下載預訓練模型和建立資料夾目錄結構)
- [轉換照片格式](#轉換照片格式)
- [訓練 YOLOv7 模型](#訓練-yolov7-模型)
- [訓練 YOLOv8 模型](#訓練-yolov8-模型)

--- 

- ### 下載預訓練模型和建立資料夾目錄結構

1. clone YOLOv7 並下載預訓練的 YOLOv7 模型(權重)：

    ```bash
    !git clone https://github.com/WongKinYiu/yolov7.git
    !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
    ```

2. 建立訓練、驗證和測試資料集所需的資料夾目錄結構：

    ```python
    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train/images')
    create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train/labels')
    create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/val/images')
    create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/val/labels')
    create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images')
    create_folder('/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/labels')
    ```

- ### 轉換照片格式

1. 將 PASCAL VOC XML 轉換為 YOLO 格式：

```python
def xml_to_yolo_bbox(bbox, w, h):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]
```

2. 將image和label移到訓練、驗證和測試資料夾中：

```python
img_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/images'
_, _, files = next(os.walk(img_folder))
pos = 0
for f in files:
    source_img = os.path.join(img_folder, f)
    if pos < 700:
        dest_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train'
    elif pos >= 700 and pos < 800:
        dest_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/val'
    else:
        dest_folder = '/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test'
    destination_img = os.path.join(dest_folder, 'images', f)
    shutil.copy(source_img, destination_img)

    label_file_basename = os.path.splitext(f)[0]
    label_source_file = f"{label_file_basename}.xml"
    label_dest_file = f"{label_file_basename}.txt"

    label_source_path = os.path.join('/content/drive/MyDrive/ObjectDetection_Yolo7/annotations', label_source_file)
    label_dest_path = os.path.join(dest_folder, 'labels', label_dest_file)

    if os.path.exists(label_source_path):
        tree = ET.parse(label_source_path)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        result = []

        for obj in root.findall('object'):
            label = obj.find("name").text
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")
            if result:
                with open(label_dest_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(result))

    pos += 1
```

- ### 訓練 YOLOv7 模型

1. 訓練 YOLOv7 模型：

```bash
!python /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/train.py --weights 'yolov7-e6e.pt' --data /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/data/masks.yaml --workers 1 --batch-size 4 --img 416 --cfg /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/cfg/training/yolov7-masks.yaml --name yolov7 --epochs 10
```

2. 使用訓練好的 YOLOv7 模型進行偵測：

```bash
!python /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/detect.py --weights /content/runs/train/yolov7/weights/best.pt --conf 0.4 --img-size 640 --source /content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images
```

3. 顯示偵測結果：

```python
import glob
from IPython.display import Image, display

i = 0
limit = 1  # 最大顯示圖片數量
for imageName in glob.glob('/content/runs/detect/exp/*.png'):
    if i < limit:
        display(Image(filename=imageName))
        print("\n")
    i += 1
```

- ### 訓練 YOLOv8 模型

1. 安裝 ultralytics

```bash
!pip install ultralytics
```

2. 使用 YOLOv8 的預訓練模型：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
result = model("/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images/maksssksksss89.png")
print(result)

!yolo detect predict model=yolov8n.pt source="/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images/maksssksksss89.png" conf=0.3
```

3. 顯示 YOLOv8 偵測結果：

```python
display(Image(filename="/content/runs/detect/predict/maksssksksss89.png"))
```

4. 訓練 YOLOv8 模型：

```python
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

results = model.train(data="/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/data/masks.yaml", epochs=1, imgsz=512, batch=4, verbose=True, device='gpu')

model.export()
```

5. 使用訓練好的 YOLOv8 模型進行偵測：

```bash
!yolo detect predict model=/content/runs/detect/train/weights/best.pt source="/content/drive/MyDrive/ObjectDetection_Yolo7/yolov7/test/images/maksssksksss89.png" conf=0.4
```

6. 顯示偵測結果：

```python
display(Image(filename="/content/runs/detect/predict2/maksssksksss89.png"))
```

![image](https://github.com/YU-TING-HU/Image_Audio_ObjectDetection/assets/169147511/b727b45f-befa-47d2-a1a3-f4ea5394d572)

---

# Audio Classification with CNN

使用卷積神經網路(CNN)做音訊分類，辨識心跳聲是否異常。

## 目錄

- [套件](#使用套件)
- [分析流程](#資料分析流程)

## 使用套件

- `torch`
- `torchaudio`
- `seaborn`
- `matplotlib`
- `numpy`
- `sklearn`

```sh
pip install torch torchaudio seaborn matplotlib numpy scikit-learn
```

## 資料分析流程

1. [**資料探勘**：](#1-資料探勘)呈現音訊檔案的波形圖和頻譜圖。
2. [**資料預處理**：](#2-資料預處理)將音訊檔案轉換為頻譜圖圖片，並拆分為模型的訓練集和測試集。
3. [**CNN模型**：](#3-CNN模型)建構一個可以多類別分類圖片的卷積神經網路(CNN for multi-class image classification)。
4. [**模型訓練**：](#4-模型訓練)訓練CNN模型。
5. [**模型測試**：](#5-模型測試)評估模型分類結果。

### 1. 資料探勘

呈現音訊檔案的波形圖和頻譜圖結構。

```python
import torchaudio
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)

def plot_specgram(waveform, sample_rate, file_path = 'test2.png'):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    fig, axes = plt.subplots(num_channels, 1)
    fig.set_size_inches(10, 10)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    plt.gca().set_axis_off()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0)

wav_file = '/content/drive/MyDrive/065_CNN_AudioClassification/set_a/extrahls__201101070953.wav'
data_waveform, sr = torchaudio.load(wav_file)
data_waveform.size()

plot_waveform(data_waveform, sample_rate=sr)

spectogram = torchaudio.transforms.Spectrogram()(data_waveform)
spectogram.size()
plot_specgram(waveform=data_waveform, sample_rate=sr)
```

### 2. 資料預處理

音訊檔案被轉換為頻譜圖圖片，並被拆分為訓練集和測試集，作為 CNN 的 input。

```python
import torchaudio
import os
import random

wav_path = '/content/drive/MyDrive/065_CNN_AudioClassification/set_a'
wav_filenames = os.listdir(wav_path)
random.shuffle(wav_filenames)

ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']
for f in wav_filenames:
    class_type = f.split('_')[0]
    f_index = wav_filenames.index(f)
    target_path = 'train' if f_index < 140 else 'test'
    class_path = f"{target_path}/{class_type}"
    file_path = f"{wav_path}/{f}"
    f_basename = os.path.basename(f)
    f_basename_wo_ext = os.path.splitext(f_basename)[0]
    target_file_path = f"{class_path}/{f_basename_wo_ext}.png"
    if (class_type in ALLOWED_CLASSES):
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        data_waveform, sr = torchaudio.load(file_path)
        plot_specgram(waveform=data_waveform, sample_rate=sr, file_path=target_file_path)
```

### 3. CNN模型

建構用於多類別分類圖片的卷積神經網路(CNN for multi-class image classification)，對頻譜圖圖片進行分類。

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.Resize((100,100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

CLASSES = ['artifact', 'extrahls', 'murmur', 'normal']
NUM_CLASSES = len(CLASSES)

class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size= 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels= 16, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100*100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

model = ImageMulticlassClassificationNet()
```

### 4. 模型訓練

使用訓練集訓練 CNN，並使用 Adam 優化模型。

```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
losses_epoch_mean = []
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    losses_epoch = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses_epoch.append(loss.item())
    losses_epoch_mean.append(np.mean(losses_epoch))
    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {np.mean(loss

es_epoch):.4f}')

sns.lineplot(x=list(range(len(losses_epoch_mean))), y=losses_epoch_mean)
```

### 5. 模型測試

在測試集上使用準確率、混淆矩陣評估模型。

```python
y_test = []
y_test_hat = []

for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()

    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')

cm = confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
```
