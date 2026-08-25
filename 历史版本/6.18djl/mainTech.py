from machine import UART
import sensor,math,time

class myUART:
    def __init__(self,uart_id,baudrate=115200):
        self.uart=UART(uart_id,baudrate=baudrate)

    def readFlag(self,byteFlag):
        assert byteFlag in (b'O',b'P'),"we only have b'O' and b'P' 2 type flags"
        uart_num = self.uart.any()
        if uart_num==0:
            return False
        uart_str = self.uart.read(uart_num)
        if not(uart_str and (byteFlag in uart_str)):
            return False
        return True

    def writePmode(self,num0,num1):
        self.uart.write("&P" + f"{num0}" + "," + f"{num1}" + "%")

    def writeMmode(self,symbolmap):
        # print('开始地图发送')
        frame = ""
        for row in symbolmap:
            for element in row:
                frame+=element
        self.uart.write("&M" + frame + "%")
        # print('地图发送完成')

    def writeErr(self):
        self.uart.write("&PERR%")
        # print('下位机发送ERR')

    def waitingSlave(self):
        print('等待下位机响应')
        while not self.readFlag(b'O'):
            continue
        # print('下位机已连接')

class Camera:
    def __init__(self,lens_corr_coeff):
        sensor.reset()
        sensor.set_brightness(200)
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)
        sensor.skip_frames(time=1000)

        self.lens_corr_coeff=lens_corr_coeff

    def capture(self,corners=None):
        if corners==None:
            return sensor.snapshot().lens_corr(self.lens_corr_coeff)
        return sensor.snapshot().lens_corr(self.lens_corr_coeff).rotation_corr(corners=corners)

    @staticmethod
    def getWidthHeight():
        return sensor.width(),sensor.height()

class GameSymbol:
    def __init__(self):
        self.car_angle=None
        self.perspective_matrix=None
        self.static_map = []

        self.grid_base ={
            "cell_size_x": 0,
            "cell_size_y": 0,
            "grid_min_x": 0,
            "grid_min_y": 0,
            }

    @staticmethod
    def detect_and_draw_wall_border(img,roi):
        """
        识别外墙壁的最小包围矩形，并绘制边框
        :param img: 摄像头帧
        :return: 墙壁矩形 (min_x, min_y, max_x, max_y) 或 None
        """

        # 步骤1：检测所有墙壁色块（LAB模式）
        wall_blobs = img.find_blobs(
            [ELEMENT["wall"]["threshold"]],
            roi=roi,                # 主要获取区域
            pixels_threshold=300,   # 过滤小噪声色块
            area_threshold=300,     # 过滤小面积噪声
            merge=True,             # 合并所有相邻墙壁色块，便于找外轮廓
            lab=True,               #
        )

        if not wall_blobs:
            print("未检测到墙壁色块！")
            return None

        # 步骤2：计算所有墙壁色块的最小包围矩形（外边框）
        min_x = WIDTH   # 初始化为画面最右x
        min_y = HEIGHT  # 初始化为画面最下y
        max_x = 0                # 初始化为画面最左x
        max_y = 0                # 初始化为画面最上y

        for b in wall_blobs:
            # 更新外边框的最小/最大坐标
            blob_left = b.x()
            blob_top = b.y()
            blob_right = b.x() + b.w()
            blob_bottom = b.y() + b.h()

            min_x = min(min_x, blob_left)
            min_y = min(min_y, blob_top)
            max_x = max(max_x, blob_right)
            max_y = max(max_y, blob_bottom)

        # 步骤3：边界校验（避免超出摄像头画面范围）
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(WIDTH - 1, max_x)
        max_y = min(HEIGHT - 1, max_y)

        # 步骤4：校验矩形有效性（宽高需大于20像素，过滤无效小色块）
        wall_width = max_x - min_x
        wall_height = max_y - min_y
        if wall_width < 20 or wall_height < 20:
            print("检测到的墙壁区域过小，无效！")
            return None

        # 步骤5：绘制外墙壁边框（核心功能）
        # img.draw_rectangle(
        #     min_x,          # 矩形左上角x
        #     min_y,          # 矩形左上角y
        #     wall_width,     # 矩形宽度
        #     wall_height,    # 矩形高度
        #     color=(255,0,0),
        #     thickness=2
        # )
        co1 = (min_x+1,min_y+1)
        co2 = (max_x-1,min_y+1)
        co3 = (max_x-1,max_y-1)
        co4 = (min_x+1,max_y-1)
        return [co1,co2,co3,co4]

    def detect_and_draw_car(self,img,map_corners):
        # 如果传入了新角点，更新透视矩阵
        if map_corners:
            self.perspective_matrix = self.compute_homography(map_corners)

        if self.perspective_matrix is None:
            return None, None

        head_blobs = img.find_blobs([ELEMENT["car_head"]["threshold"]], roi=ROI_CAR, pixels_threshold=100, area_threshold=100, merge=True, margin=0)
        back_blobs = img.find_blobs([ELEMENT["car_back"]["threshold"]], roi=ROI_CAR, pixels_threshold=100, area_threshold=100, merge=True, margin=0)

        if not head_blobs or not back_blobs:
            # print("未检测到小车！")
            self.car_angle = None
            return None, None
        b1 = max(head_blobs, key=lambda b: b.pixels())   # 车头
        b2 = max(back_blobs, key=lambda b: b.pixels())   # 车尾

        # 获取在原始畸变图像中的像素坐标
        x1, y1 = b1.cx(), b1.cy()
        x2, y2 = b2.cx(), b2.cy()

        # 核心：将车头和车尾分别映射到 0~1 的归一化地图空间
        nx1, ny1 = self.apply_homography(self.perspective_matrix, x1, y1)
        nx2, ny2 = self.apply_homography(self.perspective_matrix, x2, y2)

        # 乘以地图网格数，直接得到真实世界的虚拟坐标 (完美矫正)
        ax1 = nx1 * GRID_X_NUM
        ay1 = ny1 * GRID_Y_NUM
        ax2 = nx2 * GRID_X_NUM
        ay2 = ny2 * GRID_Y_NUM

        # 1. 小车坐标为真实世界车头车尾的中点
        ax = (ax1 + ax2) / 2 - 0.5
        ay = (ay1 + ay2) / 2 - 0.5

        ax, ay = self.radial_distortion_correct(ax, ay)

        # 2. 计算完美矫正后的车身角度 (用真实空间下的相对位置算角度)
        dx = ax2 - ax1
        dy = ay2 - ay1

        if dx == 0 and dy == 0:
            self.car_angle = None
            return (round(ax, 2), round(ay, 2)),self.car_angle
        self.car_angle = math.degrees(math.atan2(-dx, dy))
        if self.car_angle > 180:
            self.car_angle -= 360
        elif self.car_angle < -180:
            self.car_angle += 360
        self.car_angle = round(self.car_angle,2)
        return (round(ax, 2), round(ay, 2)),self.car_angle

    def detect_grid_color_and_generate_map(self,img):#检验矩形区域，分割识别
        max_x = WIDTH   # 初始化为画面最右x
        max_y = HEIGHT  # 初始化为画面最下y
        min_x = 0                # 初始化为画面最左x
        min_y = 0                # 初始化为画面最上y
        wall_w = max_x - min_x
        wall_h = max_y - min_y

        # 1. 计算正方形ROI格子尺寸
        x_cell_size = wall_w / GRID_X_NUM
        y_cell_size = wall_h / GRID_Y_NUM

        self.grid_base["grid_min_x"] = min_x
        self.grid_base["grid_min_y"] = min_y
        # self.grid_base["cell_size_x"] = x_cell_size
        # self.grid_base["cell_size_y"] = y_cell_size

        # 2. 初始化符号地图(全为地面)
        self.static_map = [[ELEMENT["ground"]["symbol"] for _ in range(GRID_X_NUM)] for _ in range(GRID_Y_NUM)]

        # 3. 遍历每个格子ROI
        for y_idx in range(GRID_Y_NUM):
            for x_idx in range(GRID_X_NUM):
                # 定义当前格子ROI
                roi_x = int(min_x + x_idx * x_cell_size)
                roi_y = int(min_y + y_idx * y_cell_size)
                roi_w = int(x_cell_size)
                roi_h = int(y_cell_size)

                # 边界校验
                roi_x = max(0, roi_x)
                roi_y = max(0, roi_y)
                roi_w = min(roi_w, WIDTH - roi_x)
                roi_h = min(roi_h, HEIGHT - roi_y)
                roi_total_pixels = roi_w * roi_h  # ROI总像素数
                if roi_total_pixels == 0:
                    continue

                # ========== 核心：统计每个元素在ROI内的像素占比 ==========
                element_ratio = {}  # 存储元素:占比
                for element in ELEMENT_PRIORITY:
                    blobs = img.find_blobs(
                        [ELEMENT[element]["threshold"]],
                        roi=(roi_x, roi_y, roi_w, roi_h),
                        pixels_threshold=90,
                        # area_threshold=200,
                        merge=True,
                    )
                    # 计算该元素在ROI内的总像素数
                    total_pixels = sum([b.pixels() for b in blobs])
                    # 计算占比
                    ratio = total_pixels / roi_total_pixels
                    element_ratio[element] = ratio

                # ========== 选择占比最高的有效元素 ==========
                max_ratio = 0
                current_element = "ground"
                for element in ELEMENT_PRIORITY:
                    ratio = element_ratio[element]
                    # 条件：占比>当前最大值 + 占比≥最小阈值
                    if ratio > max_ratio:
                        max_ratio = ratio
                        current_element = element
                # ========== 更新地图 + 可视化 ==========
                self.static_map[y_idx][x_idx] = ELEMENT[current_element]["symbol"]
        return self.static_map.copy()

    def calc_virtual_coord_2(self,detected_actual):
        """
        将摄像头像素坐标转换为虚拟地图坐标。
        """
        if detected_actual is None:
            return None

        x, y = detected_actual

        cell_x = self.grid_base["cell_size_x"]
        cell_y = self.grid_base["cell_size_y"]

        if cell_x == 0 or cell_y == 0:
            return None

        base_virtual = (0, 0)
        base_actual = (cell_x // 2, cell_y // 2)

        dx_actual = detected_actual[0] - base_actual[0]
        dy_actual = detected_actual[1] - base_actual[1]

        dx_virtual = dx_actual / cell_x
        dy_virtual = dy_actual / cell_y

        virtual_x = base_virtual[0] + dx_virtual
        virtual_y = base_virtual[1] + dy_virtual

        # ==========================
        # 径向畸变矫正
        # ==========================
        virtual_x, virtual_y = self.radial_distortion_correct(virtual_x, virtual_y)
        return (round(virtual_x, 2), round(virtual_y, 2))

    @staticmethod
    def compute_homography(corners):
        """根据四个角点计算透视映射逆矩阵"""
        # corners: [左上, 右上, 右下, 左下]
        u0, v0 = corners[0]
        u1, v1 = corners[1]
        u2, v2 = corners[2]
        u3, v3 = corners[3]

        du1 = u1 - u2
        du2 = u3 - u2
        su = u0 - u1 + u2 - u3
        dv1 = v1 - v2
        dv2 = v3 - v2
        sv = v0 - v1 + v2 - v3

        den = (du1 * dv2) - (dv1 * du2)
        if den == 0:
            g = 0
            h = 0
        else:
            g = (su * dv2 - sv * du2) / den
            h = (du1 * sv - dv1 * su) / den

        a = u1 * (g + 1) - u0
        b = u3 * (h + 1) - u0
        c = u0
        d = v1 * (g + 1) - v0
        e = v3 * (h + 1) - v0
        f = v0

        # 伴随矩阵元素 (直接用于点映射)
        A = e - f * h
        B = c * h - b
        C = b * f - c * e
        D = f * g - d
        E = a - c * g
        F = c * d - a * f
        G = d * h - e * g
        H_m = b * g - a * h
        I_m = a * e - b * d

        return (A, B, C, D, E, F, G, H_m, I_m)

    @staticmethod
    def apply_homography(matrix, x, y):
        """应用透视矩阵，返回归一化坐标 (0~1)"""
        A, B, C, D, E, F, G, H_m, I_m = matrix
        w = G * x + H_m * y + I_m
        if w == 0:
            w = 0.0001
        x_norm = (A * x + B * y + C) / w
        y_norm = (D * x + E * y + F) / w
        return x_norm, y_norm

    @staticmethod
    def get_perspective_corners(img,roi):
        """
        使用最小外接旋转矩形获取完美的四个角点，并按顺时针(TL, TR, BR, BL)排序
        """
        wall_blobs = img.find_blobs(
            [ELEMENT["wall"]["threshold"]],
            roi=roi,
            pixels_threshold=300,
            area_threshold=300,
            merge=True,   # 合并为一个大墙壁
            lab=True
        )

        if not wall_blobs:
            return None

        # 取面积最大的那个色块（过滤掉可能没有被 merge 的小噪点）
        main_wall = max(wall_blobs, key=lambda b: b.pixels())

        # 1. 核心：直接获取最小外接旋转矩形的四个角点！
        # 这是一个完美的数学矩形，不会受到缺角、毛刺的影响
        corners = main_wall.min_corners()

        # 2. 计算几何中心点
        center_x = sum([p[0] for p in corners]) / 4
        center_y = sum([p[1] for p in corners]) / 4

        # 3. 按左右切分
        left_points = [p for p in corners if p[0] < center_x]
        right_points = [p for p in corners if p[0] >= center_x]

        # 安全兜底：如果中心点切分失败（极少见），按 x 坐标强切
        if len(left_points) != 2 or len(right_points) != 2:
            corners_list = list(corners)
            corners_list.sort(key=lambda p: p[0])
            left_points = corners_list[:2]
            right_points = corners_list[2:]

        # 4. 在左右各自按 y 坐标排序（y小在上，y大在下）
        left_points.sort(key=lambda p: p[1])
        tl = left_points[0]  # 左上
        bl = left_points[1]  # 左下

        right_points.sort(key=lambda p: p[1])
        tr = right_points[0] # 右上
        br = right_points[1] # 右下

        # 返回供 rotation_corr 使用的顺序
        return [tl, tr, br, bl]

    @staticmethod
    def radial_distortion_correct(x,y):
        """
        四方向分段二次坐标矫正。
        公式：
            x_correct = cx + dx * (1 + kx * nx^2)
            y_correct = cy + dy * (1 + ky * ny^2)
        """

        cx = DISTORT_CENTER_X
        cy = DISTORT_CENTER_Y

        half_x = DISTORT_CENTER_X
        half_y = DISTORT_CENTER_Y

        if half_x == 0 or half_y == 0:
            return x, y

        dx = x - cx
        dy = y - cy

        nx = dx / half_x
        ny = dy / half_y

        # 根据当前位置选择不同方向的矫正参数
        if dx < 0:
            kx = KX_LEFT
        else:
            kx = KX_RIGHT

        if dy < 0:
            ky = KY_UP
        else:
            ky = KY_DOWN

        corrected_x = cx + dx * (1 + kx * nx * nx)
        corrected_y = cy + dy * (1 + ky * ny * ny)

        # 防止矫正后越界
        if corrected_x < 0:
            corrected_x = 0
        elif corrected_x > GRID_X_NUM - 1:
            corrected_x = GRID_X_NUM - 1

        if corrected_y < 0:
            corrected_y = 0
        elif corrected_y > GRID_Y_NUM - 1:
            corrected_y = GRID_Y_NUM - 1

        return corrected_x, corrected_y

def print_symbol_map(symbol_map):
    """
    格式化打印符号地图。
    """
    if symbol_map is None:
        return None

    print("\n===== 游戏地图 =====")
    for row in symbol_map:
        row_str = "".join(row)
        print(row_str)
    print("===================================\n")

# ==========================
# 串口初始化
# ==========================
myuart = myUART(12)
#myuart.waitingSlave()#等待下位机
# ==========================
# 摄像头初始化
# ==========================
camera=Camera(0.35)
WIDTH,HEIGHT=camera.getWidthHeight()
# 这两句似乎没啥用⬇
# clock = time.clock()
# clock.tick()
# ==========================
# 颜色阈值
# ==========================
ELEMENT = {
    "wall":    {"threshold":(30, 89,  -7,  47, -79,   7),"symbol":"#"},       # 墙壁
    "box":     {"threshold":(17, 95, -42,  34,  21,  93),"symbol":"$"},      # 箱子
    "car_head":{"threshold":(65, 93, -51,  -6, -49,   2),"symbol":None},    # 车头
    "car_back":{"threshold":(63, 91, -84, -56,  -1,  86),"symbol":None},   # 车尾
    "goal":    {"threshold":(17, 88,  64, 107, -86, -29),"symbol":"."},    # 目标点
    "bomb":    {"threshold":(24, 77,  24,  95, -27,  86),"symbol":"*"},      # 炸弹
    "ground":  {"threshold":None,                        "symbol":"-"},      # 地面
    "car":     {"threshold":None,                        "symbol":"@"},   # 只作符号位
}
ROI = (0, 0, 320, 240)
ROI_CAR = (0, 0, 320, 240)
ELEMENT_PRIORITY = ["box", "goal", "bomb", "wall"]
GRID_X_NUM = 16
GRID_Y_NUM = 12
# 占比阈值
MIN_PIXEL_RATIO = 0.4
# ==========================
# 四方向分段二次畸变矫正参数
# KX_LEFT   ：调节左边区域 x 坐标
# KX_RIGHT  ：调节右边区域 x 坐标
# KY_UP     ：调节上边区域 y 坐标
# KY_DOWN   ：调节下边区域 y 坐标
# ==========================
KX_LEFT  = -0.02
KX_RIGHT = -0.02
KY_UP    = -0
KY_DOWN  = -0

DISTORT_CENTER_X = (GRID_X_NUM - 1) / 2
DISTORT_CENTER_Y = (GRID_Y_NUM - 1) / 2
# ==========================
# 初始化地图并发送
# ==========================
gamesymbol = GameSymbol()
while True:
    img = camera.capture()
    if not myuart.readFlag(b'O'):
        continue
    map_corners = gamesymbol.detect_and_draw_wall_border(img,ROI)

    if map_corners is None:
        myuart.writeErr()
        continue

    visual_ref, angle = gamesymbol.detect_and_draw_car(img, map_corners)

    if visual_ref is None:
        print("未识别到小车")
        myuart.writeErr()
        continue

    myuart.writePmode(visual_ref[0], visual_ref[1])
    # map_corners = gamesymbol.get_perspective_corners(img,ROI)
    img = camera.capture(map_corners)
    symbolmap = gamesymbol.detect_grid_color_and_generate_map(img)

    # if visual_ref is None:
    #     print("未识别到小车，不写入 @")
    #     myuart.writeMmode(symbolmap)
    #     break
    theroitic_point = (round(visual_ref[0]), round(visual_ref[1]))
    print(f"理论坐标{theroitic_point}")
    if 0 <= theroitic_point[0] < GRID_X_NUM and 0 <= theroitic_point[1] < GRID_Y_NUM:
        symbolmap[theroitic_point[1]][theroitic_point[0]] = ELEMENT["car"]["symbol"]
    else:
        print("小车理论坐标越界，不写入地图")
    myuart.writeMmode(symbolmap)
    break
# ==========================
# 主循环
# 收到P后返回小车坐标
# ==========================
# 收P发送版
while True:
    img = camera.capture()
    if not myuart.readFlag(b'P'):
        continue
    img = camera.capture()
    map_corners = gamesymbol.detect_and_draw_wall_border(img,ROI)
    # map_corners = gamesymbol.get_perspective_corners(img,ROI)
    # lcd.show_image(img, 320, 240, zoom=0)
    coord,angle = gamesymbol.detect_and_draw_car(img,map_corners)
    if coord is None:
        print("坐标识别失败")
        myuart.writeErr()
        continue
    myuart.writePmode(coord[0],coord[1])

# 一直发送版
# while True:
#     img = camera.capture()
#     map_corners = gamesymbol.detect_and_draw_wall_border(img,ROI)
#     # map_corners = gamesymbol.get_perspective_corners(img,ROI)
#     coord,angle = gamesymbol.detect_and_draw_car(img,map_corners)
#     if coord is None:
#         print("坐标识别失败")
#         myuart.writeErr()
#         continue
#     myuart.writePmode(coord[0],coord[1])
