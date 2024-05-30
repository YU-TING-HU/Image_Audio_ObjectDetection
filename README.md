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
- [訓練 YOLOv7 模型](#訓練YOLOv7模型)
- [訓練 YOLOv8 模型](#訓練YOLOv8模型)

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
