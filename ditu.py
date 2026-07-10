导入 tkinter 作为 tk
从 tkinter 导入 ttk
import numpy 作为 np
导入 cv2
从 PIL 导入 Image, ImageTk
导入 webbrowser
导入 threading
从 http.server 导入 HTTPServer, BaseHTTPRequestHandler
从socketserver 导入ThreadingMixIn
导入 json
导入 time
导入 socket
导入 urllib.request
导入 math

# --- 全局状态变量 ---
drone_state = {"lat": 34.334015, "lon": 108.99611, "height": 100.0, "pitch": -45.0}
polygon_state = {"coords": [], "action": ""}
# 34.334015 108.99611
# 前端 HTML：只保留地图和绘制功能，移除了 Canvas 投影层
MAP_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>无人机区域绘制</title>
    <style>
        html, body, #container {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
    </style>
    <!-- 引入高德地图 JS API -->
&lt;b&gt;    &lt;script src=&quot;https:&#x2F;&#x2F;webapi.amap.com&#x2F;maps?v=2.0&amp;key=yourkey&amp;plugin=AMap.MouseTool&quot;&gt;&lt;&#x2F;script&gt;&lt;&#x2F;b&gt;

<body>
    <div id="container"></div>
    <script>
        // 1. 初始化地图
        var map = new AMap.Map('container', {
            zoom: 16,
            中心：[121.4737, 31.2304],
            视图模式：'2D'
        });

        // 2. 初始化无人机标记和任务区域多边形
        window.droneMarker = new AMap.Marker({
            map: map,
            z-index: 110,
            图标: "https://a.amap.com/jsapi_demos/static/demo-center/icons/poi-marker-default.png"
        });
        
        window.taskPolygon = new AMap.Polygon({
            path: [],
            strokeColor: "#0000FF",
            strokeWeight: 3,
            fillColor: "#0000FF",
            fillOpacity: 0.2,
            map: map
        });

        // 3. 全局状态锁，防止重复触发绘制工具
        window.isDrawing = false;
        var currentMouseTool = null;

        // 4. 定时轮询后端状态
        setInterval(async () => {
            try {
                const res = await fetch('/get_state');
                const state = await res.json();

                // 更新无人机位置
                window.droneMarker.setPosition([state.drone.lon, state.drone.lat]);
                map.setCenter([state.drone.lon, state.drone.lat]);

                // 处理多边形状态
                if (state.polygon.action === 'set') {
                    window.taskPolygon.setPath(state.polygon.coords);
                    map.setFitView([window.taskPolygon]);
                } else if (state.polygon.action === 'clear') {
                    window.taskPolygon.setPath([]);
                } 
                // 激活绘制模式
                else if (state.polygon.action === 'draw' && !window.isDrawing) {
                    window.isDrawing = true; // 锁定状态
                    
                    // 清理旧的工具实例
                    if (currentMouseTool) {
                        currentMouseTool.close(true);
                    }

                    // 创建新的绘制工具
                    currentMouseTool = new AMap.MouseTool(map);
                    currentMouseTool.polygon({
                        strokeColor: "#0000FF",
                        fillColor: "#0000FF",
                        fillOpacity: 0.2
                    });

                    console.log("🗺️ 已进入绘制模式，请在地图上点击绘制多边形。");

                    // 监听绘制完成事件
                    currentMouseTool.on('draw', function(e) {
                        var path = e.obj.getPath();
                        
                        // 更新多边形显示
                        window.taskPolygon.setPath(path);
                        
                        // 关闭工具并解锁
                        currentMouseTool.close(true);
                        window.isDrawing = false;
                        
                        console.log("✅ 绘制完成，坐标已发送至后端。");
                        
                        // 将坐标发送回 Python 后端
                        fetch('/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                action: 'set_polygon',
                                coords: path
                            })
                        }).catch(err => console.error("发送坐标失败:", err));
                    });
                }

            } catch (e) {
                console.error("状态更新错误:", e);
            }
        }, 500);
    </script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MapHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/get_state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            state = {"drone": drone_state, "polygon": polygon_state}
            self.wfile.write(json.dumps(state).encode('utf-8'))
        elif self.path == '/' or self.path == '/map.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(MAP_HTML.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)

            if data.get('action') == 'set_polygon':
                polygon_state['coords'] = data['coords']
                polygon_state['action'] = 'set'
            elif data.get('action') == 'draw_polygon':
                polygon_state['action'] = 'draw'
            elif data.get('action') == 'clear_polygon':
                polygon_state['coords'] = []
                polygon_state['action'] = 'clear'

            self.send_response(200)
            self.end_headers()
        except Exception as e:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # 隐藏控制台日志


class DroneSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("无人机视角投影模拟器")
        self.root.geometry("1200x700")
        # "latitude": 34.331544,
        # "longitude": 108.99489,
        self.lat_var = tk.DoubleVar(value=34.331544)
        self.lon_var = tk.DoubleVar(value=108.99489)
        self.height_var = tk.DoubleVar(value= 491.487)
        self.pitch_var = tk.DoubleVar(value=-74.8)

        # === 左侧：控制面板 ===
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="无人机纬度:").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.lat_var).pack(fill=tk.X, pady=2)
        ttk.Label(control_frame, text="无人机经度:").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.lon_var).pack(fill=tk.X, pady=2)
        ttk.Label(control_frame, text="飞行高度 (m):").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.height_var).pack(fill=tk.X, pady=2)
        ttk.Label(control_frame, text="云台俯仰角 (°):").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.pitch_var).pack(fill=tk.X, pady=2)

        ttk.Button(control_frame, text="更新模拟", command=self.update_simulation).pack(fill=tk.X, pady=10)
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="地图多边形操作:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        ttk.Button(control_frame, text="手动绘制区域", command=self.start_draw_polygon).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="导入固定区域", command=self.import_fixed_polygon).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="清除多边形", command=self.clear_polygon).pack(fill=tk.X, pady=2)

        # === 右侧：OpenCV 无人机视角画面 ===
        self.cv_canvas = tk.Canvas(root, width=640, height=360, bg="black")
        self.cv_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 启动多线程本地服务器
        self.server = ThreadingHTTPServer(('127.0.0.1', 8765), MapHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        print("⏳ 正在启动本地地图服务器...")
        for _ in range(30):
            try:
                sock = socket.create_connection(('127.0.0.1', 8765), timeout=0.1)
                sock.close()
                print("✅ 本地服务器启动成功！")
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)

        webbrowser.open("http://127.0.0.1:8765/map.html")
        self.update_simulation()

    def project_polygon_to_image(self, poly_coords, lat, lon, height, pitch_deg, fov_deg=80, img_w=640, img_h=360):
        """
        核心透视投影：将地图上的经纬度多边形转换为 OpenCV 画面上的像素坐标
        """
        if not poly_coords or len(poly_coords) < 3:
            return []
        print(poly_coords)
        # 1. 相机内参矩阵 K
        fx = fy = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx, cy = img_w / 2.0, img_h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # 2. 相机外参旋转矩阵 R (假设仅绕 X 轴俯仰)
        pitch_rad = math.radians(pitch_deg)
        R = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ], dtype=np.float64)

        # 3. 相机平移向量 T
        T = np.array([[0], [0], [height]], dtype=np.float64)

        # 4. 将地图坐标转换为以无人机为原点的局部坐标系 (米)
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(lat))

        projected_pixels = []
        for coord in poly_coords:
            m_lon, m_lat = coord[0], coord[1]
            X = (m_lon - lon) * meters_per_deg_lon
            Y = (m_lat - lat) * meters_per_deg_lat
            Z = 0  # 地面点

            # 5. 投影公式: p = K * (R * [X,Y,Z]^T + T)
            point_cam = R @ np.array([[X], [Y], [Z]]) + T

            # 如果点在相机后方，跳过
            if point_cam[2, 0] <= 0:
                continue

            p_img = K @ point_cam
            u = int(p_img[0, 0] / p_img[2, 0])
            v = int(p_img[1, 0] / p_img[2, 0])
            projected_pixels.append([u, v])

        return np.array(projected_pixels, dtype=np.int32)
# 当前坐标34.331544,108.994889,120.101，角度-30.8
    def send_post_request(self, payload):
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request('http://127.0.0.1:8765/', data=data, method='POST',
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:
            pass

    def start_draw_polygon(self):
        print("1")
        self.send_post_request({"action": "draw_polygon"})

    def import_fixed_polygon(self):
        coords = [
            # [[108.994684, 34.336546], [108.994336, 34.334613], [108.995238, 34.334535], [108.995301, 34.336533]],
            # [[108.995941, 34.335663], [108.9958, 34.335239], [108.996416, 34.335144], [108.996505, 34.335546]]
            [[108.994671, 34.335357], [108.994682, 34.334976], [108.995456, 34.335138], [108.995444, 34.335452]]
        ]
        for coord in coords:
            self.send_post_request({"action": "set_polygon", "coords": coord})

    def clear_polygon(self):
        self.send_post_request({"action": "clear_polygon"})

    def update_simulation(self, *args):
        lat = self.lat_var.get()
        lon = self.lon_var.get()
        height = self.height_var.get()
        pitch = self.pitch_var.get()

        # 更新全局状态
        drone_state['lat'] = lat
        drone_state['lon'] = lon
        drone_state['height'] = height
        drone_state['pitch'] = pitch

        # 1. 生成 OpenCV 黑色背景画面
        img = np.zeros((360, 640, 3), dtype=np.uint8)

        # 2. 绘制十字准星
        cv2.line(img, (320, 0), (320, 360), (0, 255, 0), 1)
        cv2.line(img, (0, 180), (640, 180), (0, 255, 0), 1)
        cv2.circle(img, (320, 180), 50, (0, 255, 0), 1)

        # 3. 【核心】计算并绘制投影多边形
        pixels = self.project_polygon_to_image(polygon_state['coords'], lat, lon, height, pitch)
        if len(pixels) >= 3:
            # 绘制半透明红色填充
            overlay = img.copy()
            cv2.fillPoly(overlay, [pixels], (0, 0, 255))
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            # 绘制红色边框
            cv2.polylines(img, [pixels], isClosed=True, color=(0, 0, 255), thickness=2)

        # 4. 绘制 OSD 文本信息
        text_lines = [
            f"Lat: {lat:.4f}, Lon: {lon:.4f}",
            f"Height: {height:.1f}m, Pitch: {pitch:.1f}deg",
            f"Polygon Points: {len(polygon_state['coords'])}"
        ]
        for i, text in enumerate(text_lines):
            cv2.putText(img, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 5. 更新 Tkinter Canvas
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.cv_canvas.delete("all")
        self.cv_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.cv_canvas.image = tk_img


if __name__ == "__main__":
    root = tk.Tk()
    app = DroneSimulator(root)
    root.mainloop()
    # 34.334015 108.99611
