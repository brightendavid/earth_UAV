
import numpy as np
import cv2
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import json
import os
import subprocess
import re
from ultralytics import YOLO
from pyproj import Transformer

# ==========================================
# 1. 在这里指定你的 FFmpeg 绝对路径 (来自 jiexi.py)
# ==========================================
FFMPEG_PATH = r"E:\data\ffmpeg\bin\ffmpeg.exe"  # 请替换为你电脑上的实际路径


class DroneSimulator:
    def __init__(self, root):
        # 在类开头或 __init__ 中定义传感器物理宽度 (mm)
        self.SENSOR_WIDTH_MM = 13.2  # ⚠️ 请根据实际无人机型号修改此值！
        self.root = root
        self.root.title("无人机元数据同步模拟器")
        self._ned_transformer = None
        self._ned_origin = None  # (lat0, lon0)
        self._ned_origin_threshold = 1e-5  # ≈1m，超过此距离才重建
        # 1. 选择视频文件
        self.video_path = filedialog.askopenfilename(
            title="选择本地无人机视频 (MP4)",
            filetypes=[("Video Files", "*.mp4"), ("All Files", "*.*")]
        )
        if not self.video_path:
            return

        # 2. 【关键步骤】预解析元数据
        self.telemetry_data = self.parse_dji_doc6874(self.video_path)
        print(f"✅ 成功加载 {len(self.telemetry_data)} 帧飞行数据")

        # 3. 初始化视频流
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("❌ 无法打开视频文件")
            return

        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.h_fov, self.v_fov = 84.0, 52.0  # 相机参数

        video_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        telemetry_count = len(self.telemetry_data)
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)


        # 4. 构建 UI 界面 (滑块现在作为备用/微调使用)
        self.canvas = tk.Canvas(root, width=self.img_width // 2, height=self.img_height // 2, bg="black")
        self.canvas.pack(pady=10)

        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=10)

        # --- 状态显示 ---
        self.status_label = tk.Label(root, text="等待播放...", font=("Arial", 10), fg="blue")
        self.status_label.pack()

        # --- 备用控制滑块 (当元数据缺失时可用) ---
        tk.Label(ctrl_frame, text="高度/俯仰 (备用)").grid(row=0, column=0, columnspan=2, pady=5)

        self.height_var = tk.DoubleVar(value=100)
        self.pitch_var = tk.DoubleVar(value=-45)
        self.focal_var = tk.DoubleVar(value=8.8)  # 默认值，可根据实际机型调整

        tk.Scale(ctrl_frame, from_=0, to=500, resolution=0.1, orient=tk.HORIZONTAL,
                 label="高度 (m)", variable=self.height_var, length=200).grid(row=1, column=0)
        tk.Scale(ctrl_frame, from_=-90, to=0, resolution=0.1, orient=tk.HORIZONTAL,
                 label="俯仰 (°)", variable=self.pitch_var, length=200).grid(row=1, column=1)
        tk.Scale(ctrl_frame, from_=4, to=200, resolution=0.1, orient=tk.HORIZONTAL,
                 label="焦距 (mm)", variable=self.focal_var, length=200).grid(row=2, column=0, columnspan=2)

        # 启动主循环
        self.update_frame()

    def parse_dji_doc6874(self, video_path):
        """
        嵌入式解析器：提取大疆 MP4 中的 Doc6874 轨道
        逻辑完全基于你提供的 jiexi.py
        """
        # 确保 FFmpeg 路径有效
        ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
        if os.path.exists(FFMPEG_PATH):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

        cmd = [FFMPEG_PATH, '-i', video_path, '-map', '0:3', '-f', 'srt', '-']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                print("FFmpeg 提取文本轨道失败:", result.stderr)
                return []
        except Exception as e:
            print("FFmpeg 执行错误:", e)
            return []

        raw_text = result.stdout
        telemetry_data = []

        # 核心正则与白名单 (来自 jiexi.py)
        pattern = r'(?:\[|\s)(\w+):\s*([^\]\s]+)'
        valid_keys = {'iso', 'shutter', 'fnum', 'ev', 'color_md', 'ae_meter_md',
                      'focal_len', 'dzoom_ratio', 'latitude', 'longitude',
                      'rel_alt', 'abs_alt', 'gb_yaw', 'gb_pitch', 'gb_roll'}

        for line in raw_text .splitlines():
            line = line.strip()
            if not line:
                continue
            matches = re.findall(pattern, line)
            if matches:
                frame_data = {}
                for key, value in matches:
                    if key in valid_keys:
                        try:
                            # 尝试转为数字
                            frame_data[key] = float(value)
                        except:
                            frame_data[key] = value
                # 只有包含经纬度或高度的数据才视为有效帧
                if 'latitude' in frame_data or 'rel_alt' in frame_data:
                    telemetry_data.append(frame_data)

        return telemetry_data

    def yolo_pred(self,frame):
        # 加载模型（推荐使用ONNX/TensorRT加速）
        yolo_model = YOLO("VisDrone.pt")  # 或 yolov8n.onnx / engine
        CONF_THRESHOLD = 0.4
        results = yolo_model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        return results


    def update_frame(self):
        """ 主循环：读取视频 -> 获取元数据 -> 更新 UI -> 投影 """
        ret, frame = self.cap.read()
        orim_frame = frame.copy()


        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环
            self.update_frame()
            return

        # --- 核心逻辑：获取当前帧的元数据 ---
        current_frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 防止索引越界
        if current_frame_id <= len(self.telemetry_data):
            meta = self.telemetry_data[current_frame_id - 1]  # 列表索引从0开始
        else:
            meta = {}

        # --- 提取数据并更新 UI 变量 (自动驱动) ---
        # 优先使用元数据，如果没有则保持滑块当前值
        height = meta.get('rel_alt', self.height_var.get())
        # 【关键修复1】：大疆 gb_pitch 向下为负，你的算法可能需要正值，尝试取反
        pitch = -meta.get('gb_pitch', self.pitch_var.get())
        # 【关键修复2】：提取偏航角 (Yaw)，如果没有则默认为0
        yaw = meta.get('gb_yaw', 0.0)
        current_focal = meta.get('focal_len', self.focal_var.get())
        self.focal_var.set(current_focal)


        # 注意：大疆的 gb_pitch 定义可能与你的投影算法坐标系不同，通常需要取反或调整
        # 这里假设 gb_pitch 向下为负，符合之前的算法
        lat = meta.get('latitude', 34.331544)
        lon = meta.get('longitude', 108.99489)
        print(lat,lon)
        # 更新滑块显示值 (仅用于可视化，不影响逻辑)
        self.height_var.set(height)
        self.pitch_var.set(pitch)

        # --- 投影计算 ---
        target_polygon_wgs84 = [
            [[108.996815, 34.329921], [108.996732, 34.328705], [109.001076, 34.328688], [109.001045, 34.329878]],
            [[109.000685, 34.332031], [109.000724, 34.331544], [109.003942, 34.331713], [109.004057, 34.332455]],
            [[108.992744, 34.332165], [108.992541, 34.331479], [108.995141, 34.331635], [108.995047, 34.332178]],
            [[108.995023, 34.331398], [108.994873, 34.331297], [108.99499, 34.33126], [108.995145, 34.331366]],
            [[108.993835, 34.32979], [108.99379, 34.329376], [108.995102, 34.329395], [108.995284, 34.329639]],
            [[108.995174, 34.332462], [108.995169, 34.331589], [108.995289, 34.331602], [108.995334, 34.332479]],
            [[108.99649, 34.332176], [108.996351, 34.330969], [108.99896, 34.331027], [108.9995, 34.332162]],
            [[108.993187, 34.331474], [108.993018, 34.33064], [108.994946, 34.330787], [108.994413, 34.33149]],
            [[108.994712, 34.330338], [108.994395, 34.328786], [108.995461, 34.328776], [108.995499, 34.330139]],
            [[108.993534, 34.332146], [108.9935, 34.331555], [108.995578, 34.331934], [108.995342, 34.33214]],
            [[108.99514, 34.332997], [108.99511, 34.331939], [108.996301, 34.330936], [108.996787, 34.331657]],
            [[109.00316, 34.330485], [109.002826, 34.329151], [109.004352, 34.328908], [109.004186, 34.330517]],
            [[108.9943, 34.333757], [108.993078, 34.332525], [108.995286, 34.332497]],
            [[108.998873, 34.332277], [108.999214, 34.330517], [108.999624, 34.330573]],
            [[108.995055, 34.334063], [108.994928, 34.333138], [108.995807, 34.333231], [108.995754, 34.333697]],
            [[109.000428, 34.332277], [109.000542, 34.331218], [109.001612, 34.330667]],
            [[108.999252, 34.334765], [109.00066, 34.33325], [109.001754, 34.333671]],
            [[108.998581, 34.333617], [108.999503, 34.331898], [108.993645, 34.334857], [108.996628, 34.336753]],
            # [[108.998395, 34.333373], [108.998554, 34.331707], [109.000049, 34.331475], [108.9997, 34.333135]],
            # [[109.000299, 34.331294], [109.000207, 34.330104], [109.000974, 34.330111], [109.001167, 34.33082]],
            # [[108.994728, 34.335319], [108.994382, 34.333422], [108.99616, 34.333431], [108.995779, 34.335338]], # 接近2
            # [[108.998083, 34.335567], [108.997621, 34.334509], [108.998839, 34.33417], [108.999454, 34.335451]],
            # [[108.999745, 34.334307], [109.000132, 34.333022], [109.00165, 34.333367], [109.000633, 34.334263]],
            # [[108.99996, 34.335566], [109.000016, 34.334213], [109.000401, 34.334249], [109.000372, 34.335413]],
            # [[108.997067, 34.334087], [108.996991, 34.33358], [108.998888, 34.333831], [108.998964, 34.334282]],
            # [[108.998912, 34.333724], [108.998382, 34.332831], [108.998684, 34.332548], [108.999443, 34.333561]],
            # [[108.999208, 34.336846], [108.998006, 34.335371], [109.004506, 34.335606], [109.003889, 34.337264]],
            # [[109.00057, 34.334241], [109.000617, 34.333862], [109.002863, 34.333839], [109.002882, 34.334233]],
            # [[108.997067, 34.333073], [108.997036, 34.332471], [108.998471, 34.332503], [108.997932, 34.332816]],
            # [[109.001467, 34.332415], [109.001194, 34.331707], [108.999412, 34.331945], [109.000056, 34.332828]],
            # [[108.996171, 34.336369], [108.996421, 34.335234], [108.99189, 34.33326], [108.989093, 34.335673]],
            # [[108.994684, 34.336546], [108.994336, 34.334613], [108.995238, 34.334535], [108.995301, 34.336533]], # 接近2
            # [[108.992581, 34.33571], [108.992396, 34.333689], [108.993043, 34.333727], [108.992985, 34.334604]],
           # [[108.9941, 34.3349], [108.995056, 34.33502], [108.994412, 34.334024], [108.993508, 34.334402]],
           #  [[108.994881, 34.334581], [108.994759, 34.334179], [108.995246, 34.334226], [108.995236, 34.334488]],
           #  [[108.992894, 34.335286], [108.992011, 34.334917], [108.992125, 34.334402], [108.993487, 34.334934]],
           #  [[108.9922, 34.33203], [108.992477, 34.331458], [108.995029, 34.331754], [108.994751, 34.332411]],
           #  [[108.996886, 34.332395], [108.996735, 34.33119], [109.001778, 34.331633], [109.002014, 34.332235]],
        # [[108.994048, 34.332102], [108.993858, 34.331701], [108.995345, 34.331845], [108.995216, 34.332114]],
        #     [[108.99538, 34.335736], [108.995206, 34.334992], [108.996092, 34.334953], [108.996171, 34.335488]] ,# 2的起始位置
            [[108.99657, 34.336245], [108.996326, 34.335684], [108.998288, 34.335705], [108.998121, 34.336287]],
            [[108.991838, 34.335789], [108.991031, 34.334496], [108.993925, 34.334796], [108.993862, 34.335802]],
            # [[108.997014, 34.332129], [108.996542, 34.331491], [108.996907, 34.331066], [108.997358, 34.331615]]
            # [[108.995941, 34.335663], [108.9958, 34.335239], [108.996416, 34.335144], [108.996505, 34.335546]]
            [[108.994671, 34.335357], [108.994682, 34.334976], [108.995456, 34.335138], [108.995444, 34.335452]] # 2

        ]

        projected_pixels = self.project_polygons_to_image(
            target_polygon_wgs84, lat, lon, height, pitch, yaw, # 加入 yaw
            current_focal, self.img_width, self.img_height
        )

        # --- 绘制 OSD ---
        # 绘制中心十字
        cv2.line(frame, (self.img_width // 2 - 20, self.img_height // 2),
                 (self.img_width // 2 + 20, self.img_height // 2), (0, 255, 0), 2)
        cv2.line(frame, (self.img_width // 2, self.img_height // 2 - 20),
                 (self.img_width // 2, self.img_height // 2 + 20), (0, 255, 0), 2)

        # 状态文本
        status_text = f"Alt: {height:.1f}m | Pitch: {pitch:.1f}°"
        if meta:
            status_text += " [REAL-TIME]"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- 绘制投影多边形 ---
        # 只要多边形与画面有交集，就进行投影绘制
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for idx, pts in enumerate(projected_pixels):
            if pts.size == 0:
                continue
            color = colors[idx % len(colors)]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=color)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # ========== 3. YOLO检测 + 空间关联 【修复1】==========
        yolo_results = self.yolo_pred(orim_frame)
        detections = []
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 判断检测框中心是否在任一投影多边形内
            in_zone, zone_id = False, -1
            for z_idx, pts in enumerate(projected_pixels):
                if pts.size >= 6 and cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                    in_zone, zone_id = True, z_idx
                    break

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': "in the area",
                'conf': conf,
                'in_zone': in_zone,
                'zone_id': zone_id
            })

        # ========== 4. 绘制 ==========
        # 4a. 绘制投影多边形
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        overlay = frame.copy()  # 【性能优化】只copy一次
        for idx, pts in enumerate(projected_pixels):
            if pts.size == 0:
                continue
            color = colors[idx % len(colors)]
            cv2.fillPoly(overlay, [pts], color=color)
            cv2.polylines(frame, [pts], True, color, 2)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)  # 一次性混合

        # 4b. 绘制YOLO检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            if det['in_zone']:
                color = (0, 255, 255)  # 区域内：黄色高亮
                label = f"{det['class']} Z{det['zone_id']} {det['conf']:.2f}"
            else:
                color = (180, 180, 180)  # 区域外：灰色
                label = f"{det['class']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 显示到 Tkinter ---
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img).resize((self.img_width // 2, self.img_height // 2))
        self.tk_img = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # 下一帧
        self.root.after(33, self.update_frame)  # ~30 FPS

    def project_polygons_to_image(self, polygons_wgs84, lat, lon, height, pitch_deg, yaw_deg, fov_deg, img_w, img_h):
        """
        将多个WGS84地理坐标多边形投影到图像像素坐标
        :param polygons_wgs84: List[List[List[float]]] 多个多边形，每个多边形为 [(lon,lat), ...] 列表
        :return: List[np.ndarray] 每个元素为对应多边形的像素坐标数组 (N,2)，无效多边形返回空数组
        """
        results = []
        for poly_coords in polygons_wgs84:
            if not poly_coords or len(poly_coords) < 3:
                results.append(np.array([], dtype=np.int32))
                continue

            # 复用原有单多边形投影核心逻辑（已修复版）
            projected = self.project_polygon_to_image(
                poly_coords, lat, lon, height, pitch_deg, yaw_deg, fov_deg, img_w, img_h
            )
            results.append(projected)

        return results

    def _wgs84_to_ned(self, coords_wgs84, lat0, lon0):
        """
        精确 WGS84 -> 局部 NED 转换（带缓存）

        参数:
            coords_wgs84: shape=(N,2), 格式 [[lon, lat], ...] ← 注意是 lon,lat
            lat0, lon0:   无人机当前位置（投影原点）
        返回:
            shape=(N,3), [north_m, east_m, down_m]，down 恒为 0
        """
        # 检查是否需要重建 transformer
        need_rebuild = (
                self._ned_transformer is None or
                abs(lat0 - self._ned_origin[0]) > self._ned_origin_threshold or
                abs(lon0 - self._ned_origin[1]) > self._ned_origin_threshold
        )

        if need_rebuild:
            self._ned_transformer = Transformer.from_crs(
                "EPSG:4326",
                f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
                always_xy=True
            )
            self._ned_origin = (lat0, lon0)

        points = np.asarray(coords_wgs84, dtype=np.float64)
        east, north = self._ned_transformer.transform(points[:, 0], points[:, 1])

        return np.stack([north, east, np.zeros_like(north)], axis=-1)

    def project_polygon_to_image(self, poly_coords, lat, lon, height, pitch_deg, yaw_deg, fov_deg, img_w, img_h):
        """将地理坐标多边形投影到图像像素坐标"""
        if not poly_coords or len(poly_coords) < 3:
            return np.array([], dtype=np.int32)

        # 1. 相机内参（基于FOV计算）
        fx = fy = fov_deg * (img_w / self.SENSOR_WIDTH_MM)
        cx, cy = img_w / 2.0, img_h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)

        # 2. 经纬度 -> NED 局部坐标 (米)
        # meters_per_deg_lat = 111320.0
        # meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
        # coords = np.array(poly_coords, dtype=np.float64)

        # 【修复】poly_coords 通常为 [(lon, lat), ...]，需确认列顺序
        # 此处假设 coords[:, 0] = lon, coords[:, 1] = lat
        # north = (coords[:, 1] - lat) * meters_per_deg_lat
        # east = (coords[:, 0] - lon) * meters_per_deg_lon
        # ✅ 新代码（一行搞定）
        # 注意: target_polygon_wgs84 必须是 [[lon, lat], ...] 格式
        ned_points = self._wgs84_to_ned(poly_coords, lat, lon) # 经纬度对齐

        north = ned_points[:, 0]  # North 分量（米）
        east = ned_points[:, 1]  # East 分量（米）

        # ned_points[:, 2] 是 Down 分量，地面目标恒为 0

        down = np.full_like(north, height)  # NED中D轴向下为正，地面目标D=+height
        points_ned = np.vstack((north, east, down))  # shape: (3, N)

        # 3. 构建旋转矩阵 (NED -> OpenCV Camera)
        # 【修复】大疆Pitch向下为负，数学右手定则向下为正，在此处统一取反
        pitch_rad = math.radians(-pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # NED -> Body 旋转矩阵 (ZYX欧拉角: Yaw -> Pitch -> Roll)
        # 注意：这里仅使用Yaw和Pitch，Roll默认为0
        R_ned_to_body = np.array([
            [cos_y * cos_p, sin_y * cos_p, -sin_p],
            [-sin_y, cos_y, 0],
            [cos_y * sin_p, sin_y * sin_p, cos_p]
        ], dtype=np.float64)

        # Body -> OpenCV Camera 轴映射 (Body: X=前,Y=右,Z=下 -> Cam: X=右,Y=下,Z=前)
        R_body_to_cam = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [1, 0, 0]], dtype=np.float64)

        # 最终旋转: NED -> Camera
        R_final = R_body_to_cam @ R_ned_to_body

        # 4. 变换到相机坐标系并投影
        points_cam = R_final @ points_ned  # shape: (3, N)

        # 【安全处理】剔除相机后方及过近的点，防止除零和畸变
        valid_mask = points_cam[2, :] > 0.1
        if not np.any(valid_mask):
            return np.array([], dtype=np.int32)

        points_cam_valid = points_cam[:, valid_mask]
        p_img = K @ points_cam_valid

        # 齐次坐标转像素坐标
        u = (p_img[0, :] / p_img[2, :]).astype(np.int32)
        v = (p_img[1, :] / p_img[2, :]).astype(np.int32)

        return np.column_stack((u, v))

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneSimulator(root)
    root.mainloop()
    # 34.334207 108.996016