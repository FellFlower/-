from pyb import LED
from machine import UART
import sensor, image, math, time


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

        if b'O' in uart_str:
            return b'O'

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

    def writeErr(self, err_code):
        """
        错误帧格式：
            &E数字%

        例如：
            &E1%
            &E2%
            &E3%
        """
        self.uart.write("&E" + str(err_code) + "%")

    def waitingSlave(self):
        print("等待下位机响应")
        while not self.readFlag(b'O'):
            continue


class Camera:
    def __init__(self):
        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)

        sensor.set_brightness(200)
        sensor.skip_frames(time=200)

    def set_tag_mode(self):
        white.on()
        sensor.set_brightness(800)
        sensor.skip_frames(time=200)

    def set_normal_mode(self):
        white.off()
        sensor.set_brightness(200)
        sensor.skip_frames(time=200)

    def capture(self):
        """
        已删除 lens_corr。
        当前所有图像统一使用原始 sensor.snapshot() 坐标系。
        """
        return sensor.snapshot()

    @staticmethod
    def getWidthHeight():
        return sensor.width(), sensor.height()


class GameSymbol:
    def __init__(self):
        self.car_angle = None
        self.perspective_matrix = None
        self.norm_to_pixel_matrix = None
        self.static_map = []

        self.grid_base = {
            "cell_size_x": 0,
            "cell_size_y": 0,
            "grid_min_x": 0,
            "grid_min_y": 0,
        }

    def detect_apriltag_corners(self, img, draw=False):
        """
        使用 AprilTag 获取地图四个基准点。

        ID 0 -> 左上角，对应游戏坐标 (0, 0)
        ID 1 -> 右上角，对应游戏坐标 (15, 0)
        ID 2 -> 右下角，对应游戏坐标 (15, 11)
        ID 3 -> 左下角，对应游戏坐标 (0, 11)

        返回:
            [左上, 右上, 右下, 左下]
        """

        tags = img.find_apriltags(families=TAG_FAMILY)

        found = {}

        for tag in tags:
            tag_id = tag.id()

            if tag_id in REQUIRED_TAG_IDS:
                rect = tag.rect()
                area = rect[2] * rect[3]

                if draw:
                    img.draw_rectangle(rect, color=(255, 0, 0))
                    img.draw_cross(tag.cx(), tag.cy(), color=(0, 255, 0))

                if tag_id not in found:
                    found[tag_id] = (tag.cx(), tag.cy(), area)
                else:
                    if area > found[tag_id][2]:
                        found[tag_id] = (tag.cx(), tag.cy(), area)

        if 0 not in found or 1 not in found or 2 not in found or 3 not in found:
            return None

        return [
            (found[0][0], found[0][1]),
            (found[1][0], found[1][1]),
            (found[2][0], found[2][1]),
            (found[3][0], found[3][1]),
        ]

    def init_apriltag_map(self, max_retry=0):
        """
        初始化 AprilTag 地图坐标系。

        这个函数在主循环前执行，不等待主控发 O。

        max_retry:
            0 表示一直等待直到识别完整四个 tag。
        """

        camera.set_tag_mode()

        map_corners = None
        retry_count = 0

        while map_corners is None:
            img = camera.capture()
            map_corners = self.detect_apriltag_corners(img, draw=False)

            if map_corners is None:
                retry_count += 1

                if max_retry > 0 and retry_count >= max_retry:
                    break

                continue

        if map_corners is None:
            camera.set_normal_mode()
            return None

        self.perspective_matrix = self.compute_homography(map_corners)
        self.norm_to_pixel_matrix = self.compute_norm_to_pixel_matrix(map_corners)

        camera.set_normal_mode()

        return map_corners

    def detect_and_draw_car(self, img, map_corners):
        """
        检测小车坐标。

        当前坐标系由 AprilTag 确定：
            ID0 -> (0, 0)
            ID1 -> (15, 0)
            ID2 -> (15, 11)
            ID3 -> (0, 11)

        成功返回:
            (x, y)

        失败返回:
            None
        """

        if map_corners and self.perspective_matrix is None:
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

        # AprilTag 四角对应游戏坐标：
        # x: 0 ~ 15
        # y: 0 ~ 11
        # 所以这里使用 GRID_X_NUM - 1 和 GRID_Y_NUM - 1。
        ax1 = nx1 * (GRID_X_NUM - 1)
        ay1 = ny1 * (GRID_Y_NUM - 1)
        ax2 = nx2 * (GRID_X_NUM - 1)
        ay2 = ny2 * (GRID_Y_NUM - 1)

        ax = (ax1 + ax2) / 2
        ay = (ay1 + ay2) / 2

        # 明显越界则认为是误识别，不强行 clamp 到边界。
        if ax < -1 or ax > GRID_X_NUM or ay < -1 or ay > GRID_Y_NUM:
            self.car_angle = None
            return None

        ax, ay = self.radial_distortion_correct(ax, ay)

        return (round(ax, 2), round(ay, 2))

    def detect_grid_color_and_generate_map(self, img, map_corners):
        """
        AprilTag 版地图识别。

        四个 Tag 刚好遮住四个外墙角块。
        因此：
            1. 外圈墙块直接固定为 #
            2. 只检测内部 10×14：
                x = 1 ~ 14
                y = 1 ~ 10
        """

        if map_corners is None:
            return None

        if self.norm_to_pixel_matrix is None:
            self.norm_to_pixel_matrix = self.compute_norm_to_pixel_matrix(map_corners)

        self.static_map = [
            [ELEMENT["ground"]["symbol"] for _ in range(GRID_X_NUM)]
            for _ in range(GRID_Y_NUM)
        ]

        # 外圈墙块直接固定为 #
        for y in range(GRID_Y_NUM):
            for x in range(GRID_X_NUM):
                if x == 0 or x == GRID_X_NUM - 1 or y == 0 or y == GRID_Y_NUM - 1:
                    self.static_map[y][x] = ELEMENT["wall"]["symbol"]

        # 内部 10×14 检测
        for y_idx in range(1, GRID_Y_NUM - 1):
            for x_idx in range(1, GRID_X_NUM - 1):

                roi = self.get_cell_roi_by_tag_coord(x_idx, y_idx)

                if roi is None:
                    continue

                roi_x, roi_y, roi_w, roi_h = roi
                roi_total_pixels = roi_w * roi_h

                if roi_total_pixels <= 0:
                    continue

                element_ratio = {}

                for element in ELEMENT_PRIORITY:
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

                if max_ratio < MIN_PIXEL_RATIO:
                    current_element = "ground"

                self.static_map[y_idx][x_idx] = ELEMENT[current_element]["symbol"]

        return [row[:] for row in self.static_map]

    def get_cell_roi_by_tag_coord(self, x_idx, y_idx):
        """
        根据 AprilTag 坐标系反推出某个内部格子的 ROI。

        只用于内部:
            x_idx = 1 ~ 14
            y_idx = 1 ~ 10
        """

        gx0 = x_idx - CELL_SAMPLE_HALF
        gx1 = x_idx + CELL_SAMPLE_HALF
        gy0 = y_idx - CELL_SAMPLE_HALF
        gy1 = y_idx + CELL_SAMPLE_HALF

        p1 = self.game_to_pixel(gx0, gy0)
        p2 = self.game_to_pixel(gx1, gy0)
        p3 = self.game_to_pixel(gx1, gy1)
        p4 = self.game_to_pixel(gx0, gy1)

        if p1 is None or p2 is None or p3 is None or p4 is None:
            return None

        xs = [p1[0], p2[0], p3[0], p4[0]]
        ys = [p1[1], p2[1], p3[1], p4[1]]

        min_x = int(min(xs))
        max_x = int(max(xs))
        min_y = int(min(ys))
        max_y = int(max(ys))

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(WIDTH - 1, max_x)
        max_y = min(HEIGHT - 1, max_y)

        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w <= 0 or roi_h <= 0:
            return None

        return (min_x, min_y, roi_w, roi_h)

    def game_to_pixel(self, gx, gy):
        """
        游戏坐标 -> 图像像素坐标。

        AprilTag 四个点对应：
            (0,0), (15,0), (15,11), (0,11)

        因此：
            nx = gx / 15
            ny = gy / 11
        """

        if self.norm_to_pixel_matrix is None:
            return None

        nx = gx / (GRID_X_NUM - 1)
        ny = gy / (GRID_Y_NUM - 1)

        return self.apply_projective(self.norm_to_pixel_matrix, nx, ny)

    @staticmethod
    def compute_homography(corners):
        """
        根据四个点计算：图像像素坐标 -> 归一化坐标 的逆矩阵。
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
    def compute_norm_to_pixel_matrix(corners):
        """
        根据四个点计算：归一化坐标 -> 图像像素坐标 的正向矩阵。
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

        return (a, b, c, d, e, f, g, h, 1)

    @staticmethod
    def apply_projective(matrix, x, y):
        A, B, C, D, E, F, G, H_m, I_m = matrix

        w = G * x + H_m * y + I_m

        if w == 0:
            w = 0.0001

        out_x = (A * x + B * y + C) / w
        out_y = (D * x + E * y + F) / w

        return out_x, out_y

    @staticmethod
    def apply_homography(matrix, x, y):
        """
        图像像素坐标 -> 归一化坐标。
        """
        return GameSymbol.apply_projective(matrix, x, y)

    @staticmethod
    def radial_distortion_correct(x, y):
        """
        四方向分段二次坐标矫正。
        注意：这不是 lens_corr，保留它是为了兼容你原来的坐标微调参数。
        如果 AprilTag 坐标已经够准，可以把 KX/KY 全部改成 0。
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
    直接使用初始化阶段识别到的 AprilTag 坐标系，不再重新找外框。
    """
    global map_corners

    if map_corners is None:
        return False

    img = camera.capture()

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
# 摄像头与 LED 初始化
# ==========================
white = LED(4)
camera = Camera()
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
        "threshold": (69, 96, -63, -15, -56, 1),
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
ROI_CAR = (0, 0, 320, 240)

ELEMENT_PRIORITY = ["box", "goal", "bomb", "wall"]

GRID_X_NUM = 16
GRID_Y_NUM = 12

MIN_PIXEL_RATIO = 0.4

DEBUG_PRINT_MAP = False

TAG_FAMILY = image.TAG16H5
REQUIRED_TAG_IDS = (0, 1, 2, 3)

# 内部格子采样半径
# 0.5 是完整格子，0.35 更偏向中心区域，减少边界串色
CELL_SAMPLE_HALF = 0.35


# ==========================
# 错误码说明
# 1：AprilTag 初始化失败
# 2：map_corners 为空，也就是没有可用地图角点
# 3：O 模式下小车识别失败
# 4：O 模式下地图识别失败
# 5：P 模式下单次坐标识别失败
# 6：O 模式下小车理论坐标越界，不能写入地图
# ==========================
ERR_TAG_INIT_FAIL = 1
ERR_NO_MAP_CORNERS = 2
ERR_CAR_DETECT_FAIL = 3
ERR_MAP_DETECT_FAIL = 4
ERR_COORD_DETECT_FAIL = 5
ERR_CAR_OUT_OF_RANGE = 6


# ==========================
# 周期性发送坐标参数
# ==========================
COORD_SEND_INTERVAL_MS = 2000
auto_send_coord = False
last_coord_send_ms = 0


# ==========================
# 四方向分段二次坐标矫正参数
# ==========================
KX_LEFT = -0.02
KX_RIGHT = -0.0
KY_UP = -0.0
KY_DOWN = -0.0

DISTORT_CENTER_X = (GRID_X_NUM - 1) / 2
DISTORT_CENTER_Y = (GRID_Y_NUM - 1) / 2


# ==========================
# AprilTag 初始化
# 不等待主控发 O，上电/运行后立即完成。
# 完成后关闭白灯，亮度恢复 200。
# ==========================
gamesymbol = GameSymbol()
map_corners = gamesymbol.init_apriltag_map()

if map_corners is None:
    myuart.writeErr(ERR_TAG_INIT_FAIL)


# ==========================
# 主程序
# O：使用已初始化的 AprilTag 坐标系，识别地图 + 返回初始坐标 + 返回地图 + 开启周期坐标发送
# P：返回当前坐标一次
# S：停止周期坐标发送
# ==========================
while True:
    img = camera.capture()
    cmd = myuart.readCmd()

    if cmd == b'O':
        if map_corners is None:
            myuart.writeErr(ERR_NO_MAP_CORNERS)
            continue

        visual_ref = gamesymbol.detect_and_draw_car(img, map_corners)

        if visual_ref is None:
            # print("未识别到小车")
            myuart.writeErr(ERR_CAR_DETECT_FAIL)
            continue

        myuart.writePmode(visual_ref[0], visual_ref[1])

        symbolmap = gamesymbol.detect_grid_color_and_generate_map(img, map_corners)

        if symbolmap is None:
            # print("地图识别失败")
            myuart.writeErr(ERR_MAP_DETECT_FAIL)
            continue

        theroitic_point = (round(visual_ref[0]), round(visual_ref[1]))

        if 0 <= theroitic_point[0] < GRID_X_NUM and 0 <= theroitic_point[1] < GRID_Y_NUM:
            symbolmap[theroitic_point[1]][theroitic_point[0]] = ELEMENT["car"]["symbol"]
        else:
            myuart.writeErr(ERR_CAR_OUT_OF_RANGE)
            continue

        # if DEBUG_PRINT_MAP:
        #     print_symbol_map(symbolmap)

        myuart.writeMmode(symbolmap)

        # auto_send_coord = True
        # last_coord_send_ms = time.ticks_ms()

        continue

    elif cmd == b'P':
        if map_corners is None:
            myuart.writeErr(ERR_NO_MAP_CORNERS)
            continue

        # img = camera.capture()

        coord = gamesymbol.detect_and_draw_car(img, map_corners)

        if coord is None:
            # print("坐标识别失败")
            myuart.writeErr(ERR_COORD_DETECT_FAIL)
            continue

        myuart.writePmode(coord[0], coord[1])
        continue

    elif cmd == b'S':
        auto_send_coord = False
        continue
    if map_corners is None:
        myuart.writeErr(ERR_NO_MAP_CORNERS)
        continue

    # img = camera.capture()

    coord = gamesymbol.detect_and_draw_car(img, map_corners)

    if coord is None:
        # print("坐标识别失败")
        myuart.writeErr(ERR_COORD_DETECT_FAIL)
        continue

    myuart.writePmode(coord[0], coord[1])
    continue


    # if auto_send_coord:
    #     now_ms = time.ticks_ms()
    #
    #     if time.ticks_diff(now_ms, last_coord_send_ms) >= COORD_SEND_INTERVAL_MS:
    #         send_current_coord_once()
    #         last_coord_send_ms = now_ms
