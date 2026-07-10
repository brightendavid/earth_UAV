import numpy as np
import cv2
import math
from PIL import Image, ImageTk
import tkinter as tk
import webbrowser

# 初始无人机状态
drone_state = {"lat": 34.331544, "lon": 108.99489, "height": 491.487, "pitch": -74.8}


class DroneSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机视觉投影模拟器 (矩阵投影版)")

        # 1. 初始化相机参数 (1920x1080, 84° x 52°)
        self.img_width, self.img_height = 1920, 1080
        self.h_fov, self.v_fov = 84.0, 52.0
        self.cx, self.cy = self.img_width / 2.0, self.img_height / 2.0

        # 2. 创建 OpenCV 画布
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # 3. 构建 UI 界面
        self.canvas = tk.Canvas(root, width=self.img_width // 2, height=self.img_height // 2, bg="black")
        self.canvas.pack(pady=10)

        slider_frame = tk.Frame(root)
        slider_frame.pack(pady=10)

        # 无人机高度滑块
        tk.Label(slider_frame, text="无人机高度 (m):").grid(row=0, column=0, padx=5)
        self.height_var = tk.DoubleVar(value=drone_state["height"])
        tk.Scale(slider_frame, from_=50, to=2000, orient=tk.HORIZONTAL,
                 variable=self.height_var, length=300, command=self.update_simulation).grid(row=0, column=1, padx=5)

        # 无人机俯仰角滑块
        tk.Label(slider_frame, text="云台俯仰角 (°):").grid(row=1, column=0, padx=5)
        self.pitch_var = tk.DoubleVar(value=drone_state["pitch"])
        tk.Scale(slider_frame, from_=-90, to=0, orient=tk.HORIZONTAL,
                 variable=self.pitch_var, length=300, command=self.update_simulation).grid(row=1, column=1, padx=5)

        # 无人机纬度滑块
        tk.Label(slider_frame, text="无人机纬度 (°):").grid(row=2, column=0, padx=5)
        self.lat_var = tk.DoubleVar(value=drone_state["lat"])
        tk.Scale(slider_frame, from_=drone_state["lat"] - 0.1, to=drone_state["lat"] + 0.1, resolution=0.0001,
                 orient=tk.HORIZONTAL,
                 variable=self.lat_var, length=300, command=self.update_simulation).grid(row=2, column=1, padx=5)

        # 无人机经度滑块
        tk.Label(slider_frame, text="无人机经度 (°):").grid(row=3, column=0, padx=5)
        self.lon_var = tk.DoubleVar(value=drone_state["lon"])
        tk.Scale(slider_frame, from_=drone_state["lon"] - 0.1, to=drone_state["lon"] + 0.1, resolution=0.0001,
                 orient=tk.HORIZONTAL,
                 variable=self.lon_var, length=300, command=self.update_simulation).grid(row=3, column=1, padx=5)

        # 状态标签
        self.status_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
        self.status_label.pack()

        # 无人机位置跳转按钮
        self.jump_btn = tk.Button(root, text="📍 在地图中查看无人机位置",
                                  font=("Arial", 12), bg="#4CAF50", fg="white",
                                  command=self.open_drone_location)
        self.jump_btn.pack(pady=10)

        self.update_simulation()

    def open_drone_location(self):
        """将当前滑块上的经纬度转换为地图URL，并在默认浏览器中打开"""
        drone_lat = self.lat_var.get()
        drone_lon = self.lon_var.get()
        map_url = f"https://uri.amap.com/marker?position={drone_lon},{drone_lat}&name=无人机当前位置&src=webapp&coordinate=gaode"
        webbrowser.open(map_url)

    def project_polygon_to_image(self, poly_coords, lat, lon, height, pitch_deg, fov_deg, img_w, img_h):
        """
        核心透视投影：将地图上的经纬度多边形转换为 OpenCV 画面上的像素坐标 (批量矩阵运算)
        """
        if not poly_coords or len(poly_coords) < 3:
            return np.array([], dtype=np.int32)

        # 1. 相机内参矩阵 K
        fx = fy = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx, cy = img_w / 2.0, img_h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # 2. 相机外参旋转矩阵 R (绕 X 轴俯仰)
        pitch_rad = math.radians(pitch_deg)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        R = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ], dtype=np.float64)

        # 3. 相机平移向量 T (无人机高度)
        T = np.array([[0], [0], [height]], dtype=np.float64)

        # 4. 将地图坐标转换为以无人机为原点的局部坐标系 (米)
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(lat))

        # 转换为 Numpy 数组以支持批量运算
        coords = np.array(poly_coords, dtype=np.float64)
        east = (coords[:, 0] - lon) * meters_per_deg_lon
        north = (coords[:, 1] - lat) * meters_per_deg_lat

        # 【关键修正】映射到相机坐标系：X_cam=东, Y_cam=0, Z_cam=北
        # 形状为 (N, 3)
        points_local = np.column_stack((east, np.zeros_like(east), north))

        # 转换为列向量形式 (3, N) 以进行矩阵乘法
        points_cam = R @ points_local.T + T

        # 5. 过滤掉相机后方的点 (Z <= 0)
        valid_mask = points_cam[2, :] > 0
        points_cam_valid = points_cam[:, valid_mask]

        if points_cam_valid.shape[1] == 0:
            return np.array([], dtype=np.int32)

        # 6. 投影到像素平面 p = K * P_cam
        p_img = K @ points_cam_valid
        u = (p_img[0, :] / p_img[2, :]).astype(np.int32)
        v = (p_img[1, :] / p_img[2, :]).astype(np.int32)

        # 组合成 (N, 2) 的像素坐标数组
        projected_pixels = np.column_stack((u, v))
        return projected_pixels

    def update_simulation(self, *args):
        height = self.height_var.get()
        pitch = self.pitch_var.get()
        drone_lat = self.lat_var.get()
        drone_lon = self.lon_var.get()

        # 定义地面上的 4 个绝对目标点 (WGS84) [lon, lat]
        target_polygon_wgs84 = [
            [108.998581, 34.333617], [108.999503, 34.331898],
            [108.993645, 34.334857], [108.996628, 34.336753]

        ]

        # 调用新的矩阵投影算法
        projected_pixels = self.project_polygon_to_image(
            target_polygon_wgs84, drone_lat, drone_lon, height, pitch,
            self.h_fov, self.img_width, self.img_height
        )

        # 重绘画布
        self.img.fill(0)

        # 绘制中心十字准星
        cv2.line(self.img, (self.img_width // 2 - 30, self.img_height // 2),
                 (self.img_width // 2 + 30, self.img_height // 2), (0, 50, 0), 1)
        cv2.line(self.img, (self.img_width // 2, self.img_height // 2 - 30),
                 (self.img_width // 2, self.img_height // 2 + 30), (0, 50, 0), 1)

        info = f"Alt: {height:.0f}m | Pitch: {pitch:.1f}° | Lat: {drone_lat:.4f} | Lon: {drone_lon:.4f}"
        cv2.putText(self.img, info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # 绘制多边形
        num_visible = len(projected_pixels)
        if num_visible == 4:
            pts = projected_pixels.reshape((-1, 1, 2))
            overlay = self.img.copy()
            cv2.fillPoly(overlay, [pts], (0, 50, 0))
            cv2.addWeighted(overlay, 0.4, self.img, 0.6, 0, self.img)
            cv2.polylines(self.img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            for px, py in projected_pixels:
                cv2.circle(self.img, (px, py), 6, (0, 0, 255), -1)
            self.status_label.config(text="✅ 多边形完全在视野内", fg="green")
        elif num_visible > 0:
            pts = projected_pixels.reshape((-1, 1, 2))
            cv2.polylines(self.img, [pts], isClosed=False, color=(0, 165, 255), thickness=3)
            self.status_label.config(text=f"⚠️ 部分超出视野 ({num_visible}/4 个顶点可见)", fg="orange")
        else:
            cv2.putText(self.img, "TARGET OUT OF VIEW", (600, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            self.status_label.config(text="❌ 多边形完全超出视野范围！", fg="red")

        # OpenCV 转 Tkinter 格式并显示
        rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).resize((self.img_width // 2, self.img_height // 2))
        self.tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneSimulator(root)
    root.mainloop()