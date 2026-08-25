from machine import UART
import sensor, math, time


class myUART:
    def __init__(self, uart_id, baudrate=115200):
        self.uart = UART(uart_id, baudrate=baudrate)

    def readFlag(self, byteFlag):
        """
        保留旧接口。
        注意：主循环里不要用 readFlag(b'O') + readFlag(b'P') 连续判断，
        因为这个函数会 read 掉串口缓冲区。
        """
        assert byteFlag in (b'O', b'P', b'S'), "we only have b'O', b'P' and b'S' 3 type flags"

        uart_num = self.uart.any()

        if uart_num == 0:
            return False

        uart_str = self.uart.read(uart_num)

        if not (uart_str and (byteFlag in uart_str)):
            return False

        return True

    def readCmd(self):
        uart_num = self.uart.any()

        if uart_num == 0:
            return None

        uart_str = self.uart.read(uart_num)

        if not uart_str:
            return None

        # 如果一次读到多个字符，优先处理 O
        if b'O' in uart_str:
            return b'O'

        # S 用于停止周期性发送坐标
        if b'S' in uart_str:
            return b'S'

        if b'P' in uart_str:
            return b'P'

        return None

    def writePmode(self, num0, num1):
        self.uart.write("&P" + f"{num0}" + "," + f"{num1}" + "%")

    def writeMmode(self, symbolmap):
        frame = ""

        for row in symbolmap:
            for element in row:
                frame += element

        self.uart.write("&M" + frame + "%")

    def writeErr(self):
        self.uart.write("&PERR%")

    def waitingSlave(self):
        print("等待下位机响应")
        while not self.readFlag(b'O'):
            continue


class Camera:
    def __init__(self, lens_corr_coeff):
        sensor.reset()
        sensor.set_brightness(200)
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)
        sensor.skip_frames(time=200)

        self.lens_corr_coeff = lens_corr_coeff

    def capture(self):
        """
        当前主流程只使用 lens_corr，不再在地图识别阶段 rotation_corr。
        """
        return sensor.snapshot().lens_corr(self.lens_corr_coeff)

    @staticmethod
    def getWidthHeight():
        return sensor.width(), sensor.height()


class GameSymbol:
    def __init__(self):
        self.car_angle = None
        self.perspective_matrix = None
        self.static_map = []

        self.grid_base = {
            "cell_size_x": 0,
            "cell_size_y": 0,
            "grid_min_x": 0,
            "grid_min_y": 0,
        }

    @staticmethod
    def detect_and_draw_wall_border(img, roi):
        """
        识别外墙壁的最小包围矩形。
        当前版本不画框。
        返回四个角点：[左上, 右上, 右下, 左下]
        """

        wall_blobs = img.find_blobs(
            [ELEMENT["wall"]["threshold"]],
            roi=roi,
            pixels_threshold=300,
            area_threshold=300,
            merge=True,
            lab=True,
        )

        if not wall_blobs:
            print("未检测到墙壁色块！")
            return None

        min_x = WIDTH
        min_y = HEIGHT
        max_x = 0
        max_y = 0

        for b in wall_blobs:
            blob_left = b.x()
            blob_top = b.y()
            blob_right = b.x() + b.w()
            blob_bottom = b.y() + b.h()

            min_x = min(min_x, blob_left)
            min_y = min(min_y, blob_top)
            max_x = max(max_x, blob_right)
            max_y = max(max_y, blob_bottom)

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(WIDTH - 1, max_x)
        max_y = min(HEIGHT - 1, max_y)

        wall_width = max_x - min_x
        wall_height = max_y - min_y

        if wall_width < 20 or wall_height < 20:
            print("检测到的墙壁区域过小，无效！")
            return None

        co1 = (min_x + 1, min_y + 1)
        co2 = (max_x - 1, min_y + 1)
        co3 = (max_x - 1, max_y - 1)
        co4 = (min_x + 1, max_y - 1)

        return [co1, co2, co3, co4]

    def detect_and_draw_car(self, img, map_corners):
        """
        检测小车坐标。
        成功返回: (x, y)
        失败返回: None
        """

        if map_corners:
            self.perspective_matrix = self.compute_homography(map_corners)

        if self.perspective_matrix is None:
            return None

        head_blobs = img.find_blobs(
            [ELEMENT["car_head"]["threshold"]],
            roi=ROI_CAR,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        back_blobs = img.find_blobs(
            [ELEMENT["car_back"]["threshold"]],
            roi=ROI_CAR,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        if not head_blobs or not back_blobs:
            self.car_angle = None
            return None

        b1 = max(head_blobs, key=lambda b: b.pixels())
        b2 = max(back_blobs, key=lambda b: b.pixels())

        x1, y1 = b1.cx(), b1.cy()
        x2, y2 = b2.cx(), b2.cy()

        nx1, ny1 = self.apply_homography(self.perspective_matrix, x1, y1)
        nx2, ny2 = self.apply_homography(self.perspective_matrix, x2, y2)

        ax1 = nx1 * GRID_X_NUM
        ay1 = ny1 * GRID_Y_NUM
        ax2 = nx2 * GRID_X_NUM
        ay2 = ny2 * GRID_Y_NUM

        ax = (ax1 + ax2) / 2 - 0.5
        ay = (ay1 + ay2) / 2 - 0.5

        ax, ay = self.radial_distortion_correct(ax, ay)

        return (round(ax, 2), round(ay, 2))

    def detect_grid_color_and_generate_map(self, img, map_corners, cell_inner_pad=3):
        """
        新地图识别逻辑：

        不使用 rotation_corr。
        直接在原始图像 img 中，根据 detect_and_draw_wall_border 得到的外框矩形，
        将矩形区域切成 16x12 个格子，然后沿用颜色分割识别。
        """

        if map_corners is None:
            print("map_corners 为空，无法生成地图")
            return None

        # map_corners: [左上, 右上, 右下, 左下]
        min_x = map_corners[0][0]
        min_y = map_corners[0][1]
        max_x = map_corners[2][0]
        max_y = map_corners[2][1]

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(WIDTH - 1, max_x)
        max_y = min(HEIGHT - 1, max_y)

        wall_w = max_x - min_x
        wall_h = max_y - min_y

        if wall_w <= 0 or wall_h <= 0:
            print("地图外框尺寸无效")
            return None

        x_cell_size = wall_w / GRID_X_NUM
        y_cell_size = wall_h / GRID_Y_NUM

        self.grid_base["grid_min_x"] = min_x
        self.grid_base["grid_min_y"] = min_y
        self.grid_base["cell_size_x"] = x_cell_size
        self.grid_base["cell_size_y"] = y_cell_size

        self.static_map = [
            [ELEMENT["ground"]["symbol"] for _ in range(GRID_X_NUM)]
            for _ in range(GRID_Y_NUM)
        ]

        for y_idx in range(GRID_Y_NUM):
            for x_idx in range(GRID_X_NUM):

                # 用边界计算，避免 int(cell_size) 累计误差
                cell_x0 = int(min_x + x_idx * x_cell_size)
                cell_y0 = int(min_y + y_idx * y_cell_size)
                cell_x1 = int(min_x + (x_idx + 1) * x_cell_size)
                cell_y1 = int(min_y + (y_idx + 1) * y_cell_size)

                roi_x = cell_x0
                roi_y = cell_y0
                roi_w = cell_x1 - cell_x0
                roi_h = cell_y1 - cell_y0

                # 轻微缩小每个格子的采样区域，减少边界串色
                if roi_w > 2 * cell_inner_pad and roi_h > 2 * cell_inner_pad:
                    roi_x += cell_inner_pad
                    roi_y += cell_inner_pad
                    roi_w -= 2 * cell_inner_pad
                    roi_h -= 2 * cell_inner_pad

                roi_x = max(0, roi_x)
                roi_y = max(0, roi_y)
                roi_w = min(roi_w, WIDTH - roi_x)
                roi_h = min(roi_h, HEIGHT - roi_y)

                roi_total_pixels = roi_w * roi_h

                if roi_total_pixels <= 0:
                    continue

                element_ratio = {}

                for element in ELEMENT_PRIORITY:
                    # 动态阈值：适配外框矩形切格后单格尺寸变化
                    pixel_th = max(5, int(roi_total_pixels * 0.05))

                    blobs = img.find_blobs(
                        [ELEMENT[element]["threshold"]],
                        roi=(roi_x, roi_y, roi_w, roi_h),
                        pixels_threshold=pixel_th,
                        merge=True,
                    )

                    total_pixels = sum([b.pixels() for b in blobs])
                    ratio = total_pixels / roi_total_pixels
                    element_ratio[element] = ratio

                max_ratio = 0
                current_element = "ground"

                for element in ELEMENT_PRIORITY:
                    ratio = element_ratio[element]

                    if ratio > max_ratio:
                        max_ratio = ratio
                        current_element = element

                self.static_map[y_idx][x_idx] = ELEMENT[current_element]["symbol"]

        return [row[:] for row in self.static_map]

    def calc_virtual_coord_2(self, detected_actual):
        """
        将摄像头像素坐标转换为虚拟地图坐标。
        当前主流程没有用到，保留备用。
        """

        if detected_actual is None:
            return None

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

        virtual_x, virtual_y = self.radial_distortion_correct(virtual_x, virtual_y)

        return (round(virtual_x, 2), round(virtual_y, 2))

    @staticmethod
    def compute_homography(corners):
        """
        根据四个角点计算透视映射逆矩阵。
        corners: [左上, 右上, 右下, 左下]
        """

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
        """
        应用透视矩阵，返回归一化坐标 0~1。
        """

        A, B, C, D, E, F, G, H_m, I_m = matrix

        w = G * x + H_m * y + I_m

        if w == 0:
            w = 0.0001

        x_norm = (A * x + B * y + C) / w
        y_norm = (D * x + E * y + F) / w

        return x_norm, y_norm

    @staticmethod
    def get_perspective_corners(img, roi):
        """
        备用函数。
        当前主流程没有使用它。
        """

        wall_blobs = img.find_blobs(
            [ELEMENT["wall"]["threshold"]],
            roi=roi,
            pixels_threshold=300,
            area_threshold=300,
            merge=True,
            lab=True
        )

        if not wall_blobs:
            return None

        main_wall = max(wall_blobs, key=lambda b: b.pixels())
        corners = main_wall.min_corners()

        center_x = sum([p[0] for p in corners]) / 4
        center_y = sum([p[1] for p in corners]) / 4

        left_points = [p for p in corners if p[0] < center_x]
        right_points = [p for p in corners if p[0] >= center_x]

        if len(left_points) != 2 or len(right_points) != 2:
            corners_list = list(corners)
            corners_list.sort(key=lambda p: p[0])
            left_points = corners_list[:2]
            right_points = corners_list[2:]

        left_points.sort(key=lambda p: p[1])
        tl = left_points[0]
        bl = left_points[1]

        right_points.sort(key=lambda p: p[1])
        tr = right_points[0]
        br = right_points[1]

        return [tl, tr, br, bl]

    @staticmethod
    def radial_distortion_correct(x, y):
        """
        四方向分段二次坐标矫正。
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
    比赛时建议关闭 DEBUG_PRINT_MAP，减少延迟。
    """

    if symbol_map is None:
        return None

    print("\n===== 游戏地图 =====")

    for row in symbol_map:
        row_str = "".join(row)
        print(row_str)

    print("===================================\n")


def send_current_coord_once():
    """
    周期性坐标发送任务使用。
    优先使用 O 阶段已经得到的 map_corners，避免每次重新识别外框。
    如果 map_corners 为空，则尝试重新识别一次外框。
    """
    global map_corners

    img = camera.capture()

    if map_corners is None:
        map_corners = gamesymbol.detect_and_draw_wall_border(img, ROI)

        if map_corners is None:
            return False

    coord = gamesymbol.detect_and_draw_car(img, map_corners)

    if coord is None:
        return False

    myuart.writePmode(coord[0], coord[1])
    return True


# ==========================
# 串口初始化
# ==========================
myuart = myUART(12)


# ==========================
# 摄像头初始化
# ==========================
camera = Camera(0.35)
WIDTH, HEIGHT = camera.getWidthHeight()


# ==========================
# 颜色阈值
# ==========================
ELEMENT = {
    "wall": {
        "threshold": (30, 89, -7, 47, -79, 7),
        "symbol": "#"
    },

    "box": {
        "threshold": (17, 95, -42, 34, 21, 93),
        "symbol": "$"
    },

    "car_head": {
        "threshold": (72, 95, -52, -17, -56, -2),
        "symbol": None
    },

    "car_back": {
        "threshold": (64, 91, -92, -52, 11, 98),
        "symbol": None
    },

    "goal": {
        "threshold": (17, 88, 64, 107, -86, -29),
        "symbol": "."
    },

    "bomb": {
        "threshold": (24, 77, 24, 95, -27, 86),
        "symbol": "*"
    },

    "ground": {
        "threshold": None,
        "symbol": "-"
    },

    "car": {
        "threshold": None,
        "symbol": "@"
    },
}


# ==========================
# 全局参数
# ==========================
ROI = (0, 7, 320, 230)
ROI_CAR = (0, 0, 320, 240)

ELEMENT_PRIORITY = ["box", "goal", "bomb", "wall"]

GRID_X_NUM = 16
GRID_Y_NUM = 12

MIN_PIXEL_RATIO = 0.4

DEBUG_PRINT_MAP = False #调试地图检测


# ==========================
# 周期性发送坐标参数
# ==========================
COORD_SEND_INTERVAL_MS = 100   # 坐标发送周期，100ms = 10Hz，可自行调整
auto_send_coord = False        # O 成功后置 True，收到 S 后置 False
last_coord_send_ms = 0


# ==========================
# 四方向分段二次畸变矫正参数
# ==========================
KX_LEFT = -0.02
KX_RIGHT = -0.02
KY_UP = -0
KY_DOWN = -0

DISTORT_CENTER_X = (GRID_X_NUM - 1) / 2
DISTORT_CENTER_Y = (GRID_Y_NUM - 1) / 2


# ==========================
# 主程序
# O：识别地图 + 返回初始坐标 + 返回地图 + 开启周期坐标发送
# P：返回当前坐标一次
# S：停止周期坐标发送
# ==========================
gamesymbol = GameSymbol()
map_corners = None


while True:
    img = camera.capture()
    cmd = myuart.readCmd()
    if cmd == b'O':
        map_corners = gamesymbol.detect_and_draw_wall_border(img, ROI)

        if map_corners is None:
            myuart.writeErr()
            continue
        visual_ref = gamesymbol.detect_and_draw_car(img, map_corners)
        if visual_ref is None:
            # print("未识别到小车")
            myuart.writeErr()
            continue
        myuart.writePmode(visual_ref[0], visual_ref[1])
        symbolmap = gamesymbol.detect_grid_color_and_generate_map(img, map_corners)
        if symbolmap is None:
            # print("地图识别失败")
            myuart.writeErr()
            continue
        theroitic_point = (round(visual_ref[0]), round(visual_ref[1]))
        print(f"理论坐标{theroitic_point}")
        if 0 <= theroitic_point[0] < GRID_X_NUM and 0 <= theroitic_point[1] < GRID_Y_NUM:
            symbolmap[theroitic_point[1]][theroitic_point[0]] = ELEMENT["car"]["symbol"]
        else:
            print("小车理论坐标越界，不写入地图")
        if DEBUG_PRINT_MAP:
            print_symbol_map(symbolmap)
        myuart.writeMmode(symbolmap)
        # O 成功完成后，开启周期性坐标发送
        auto_send_coord = True
        last_coord_send_ms = time.ticks_ms()
        continue

    elif cmd == b'P':
        img = camera.capture()
        map_corners = gamesymbol.detect_and_draw_wall_border(img, ROI)
        if map_corners is None:
            # print("地图外框识别失败")
            myuart.writeErr()
            continue
        coord = gamesymbol.detect_and_draw_car(img, map_corners)
        if coord is None:
            # print("坐标识别失败")
            myuart.writeErr()
            continue
        myuart.writePmode(coord[0], coord[1])
        continue

    elif cmd == b'S':
        auto_send_coord = False
        continue

    # # 周期性坐标发送任务
    # if auto_send_coord:
    #     now_ms = time.ticks_ms()
    #     if time.ticks_diff(now_ms, last_coord_send_ms) >= COORD_SEND_INTERVAL_MS:
    #         send_current_coord_once()
    #         last_coord_send_ms = now_ms
