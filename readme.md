# 必要环境
2.py中的路径
* ffmpeg的路径，需要在电脑上配备
FFMPEG_PATH = r"E:\data\ffmpeg\bin\ffmpeg.exe"  # 请替换为你电脑上的实际路径
* 一个对应的yolo检测模型
yolo_model = YOLO("VisDrone.pt")   或 yolov8n.onnx / engine

# 解释
## ditu.py

* 输入自己的高德地图api key.这可以在高德开发者界面免费获取。个人身份注册一个月可以15w次免费调用

```html
<!-- 引入高德地图 JS API -->
<script src="https://webapi.amap.com/maps?v=2.0&key=输入自己的key&plugin=AMap.MouseTool"></script>
```
