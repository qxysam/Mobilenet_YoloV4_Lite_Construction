## YOLOV4：目标检测模型-修改mobilenet系列主干网络及ghostnet网络-pytorch实现
## 实际项目中，采用labelme，labelimg作为数据标注工具,VOC目录采用实际类别，标注图片和标签文件进行训练

**加入letterbox_image的选项，关闭letterbox_image后网络的map一般可以得到提升。

## Libary needed-------------------------------------------
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
## 所有库更新至2021年7月，皆可运行，兼容----------------------

## 原始MAP
| 训练数据集 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| VOC07+12 | v1 VOC-Test07 | 416x416 | - | 79.72
| VOC07+12 | v2 VOC-Test07 | 416x416 | - | 80.12
| VOC07+12 | v3 VOC-Test07 | 416x416 | - | 79.01
| VOC07+12 | ghostnet VOC-Test07 | 416x416 | - | 78.69


## Notes
提供的三个权重分别是基于mobilenetv1、mobilenetv2、mobilenetv3主干网络训练而成的。使用的时候注意backbone和权重的对应。   
训练前注意修改model_path和backbone使得二者对应。   
预测前注意修改model_path和backbone使得二者对应。   


## Tricks
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。    
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。    
3、label_smoothing可用于控制是否Label Smoothing平滑。   


## 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path"        : 'model_data/yolov4_mobilenet_v2_map76.93.pth',
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "classes_path"      : 'model_data/voc_classes.txt',
    "backbone"          : 'mobilenetv2',
    
    "model_image_size"  : (416, 416, 3),
    "confidence"        : 0.5,
    "iou"               : 0.3,
    "cuda"              : True
}
3. 在predict.py里面进行设置可以进行fps测试和video视频检测。  


## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2yolo4.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
比如：classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，分类顺序和代码中的顺序保持一致，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为，和代码中的顺序保持一致：   
```python
cat
dog
...
```
8. 运行train.py即可开始训练。

## 评估步骤
评估过程可参考视频https://www.bilibili.com/video/BV1zE411u7Vw  
步骤是一样的，不需要自己再建立get_dr_txt.py、get_gt_txt.py等文件。  
1. 本文使用VOC格式进行评估。  
2. 评估前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 评估前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在评估前利用voc2yolo4.py文件生成对应的txt，评估用的txt为VOCdevkit/VOC2007/ImageSets/Main/test.txt，需要注意的是，如果整个VOC2007里面的数据集都是用于评估，那么直接将trainval_percent设置成0即可。  
5. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
6. 运行get_dr_txt.py和get_gt_txt.py，在./input/detection-results和./input/ground-truth文件夹下生成对应的txt。  
7. 运行get_map.py即可开始计算模型的mAP。
8. 更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  


## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4
