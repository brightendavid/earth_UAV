# 必要环境
2.py中的路径
* ffmpeg的路径，需要在电脑上配备
FFMPEG_PATH = r"E:\data\ffmpeg\bin\ffmpeg.exe"  # 请替换为你电脑上的实际路径
* 一个对应的yolo检测模型，这个检测模型最好是需要在小目标检测中微调过的。
yolo_model = YOLO("VisDrone.pt")   或 yolov8n.onnx / engine



# 解释
## ditu.py

* 输入自己的高德地图api key.这可以在高德开发者界面免费获取。个人身份注册一个月可以15w次免费调用

```html
<!-- 引入高德地图 JS API -->
<script src="https://webapi.amap.com/maps?v=2.0&key=输入自己的key&plugin=AMap.MouseTool"></script>
```

* 这个文件是通过调用高德api获取对应坐标地图的接口，可以简单模拟一个由4个经纬度对组成的一个矩形框，在对应的经纬度高度以及相机视角
* 目的：现在剩下的功能只有通过点击获得一个区域的四个经纬度，模拟视角的功能没有更新到最新。
## 1.py
这是一个简单的模拟，可以通过滑块，修改相机的高度和坐标

## 2.py
这是依据ffmpeg.exe解析视频元信息，这个元信息包含

> ​                    "timestamp": frame_idx / fps,  # ← 关键：按帧率生成时间戳
> ​                    "lat": frame_data["latitude"],  # ← 键名映射
> ​                    "lon": frame_data["longitude"],
> ​                    "alt": frame_data["rel_alt"],
> ​                    "roll": frame_data["gb_roll"],
> ​                    "pitch": frame_data["gb_pitch"],
> ​                    "yaw": frame_data["gb_yaw"]

* 这实际上是还包含了其他信息的，这可能和具体的相机型号和传感器有关。但是这个任务只需要以上几个参数即可
* 具体就是高度，经纬度，高度和相对起飞高度，pitch，yaw就是相机参数
* 考虑放大缩小的情况，要加入focal_len焦距。