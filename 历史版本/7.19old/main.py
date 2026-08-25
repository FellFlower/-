from pyb import LED
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
        assert byteFlag in (b'O', b'P', b'S'), \
            "we only have b'O', b'P' and b'S' 3 type flags"

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

        # 一次读到多个字符时，保持当前接口优先级：O > S > P。
        if b'O' in uart_str:
            return b'O'

        if b'S' in uart_str:
            return b'S'

        if b'P' in uart_str:
            return b'P'

        return None

    def writePmode(self, num0, num1):
        """
        坐标帧格式：
            &Px.xx,y.yy%

        固定保留两位小数，避免出现 9.510001 一类浮点尾数。
        """
        self.uart.write("&P%.2f,%.2f%%" % (num0, num1))

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

        # 保留当前版本参数：直接使用原始 snapshot，不恢复 lens_corr。
        sensor.set_brightness(200)
        # sensor.skip_frames(time=200)

    def capture(self):
        return sensor.snapshot().lens_corr(0.35)

    @staticmethod
    def getWidthHeight():
        return sensor.width(), sensor.height()


class GameSymbol:
    def __init__(self):
        self.car_angle = None
        self.perspective_matrix = None
        self.car_roi = None
        self.static_map = []

        # ==========================
        # 小车坐标轻量滤波状态
        # ==========================
        self.coord_filter_initialized = False
        self.filtered_car_x = 0.0
        self.filtered_car_y = 0.0
        self.last_raw_car_x = 0.0
        self.last_raw_car_y = 0.0
        self.jump_candidate_x = None
        self.jump_candidate_y = None
        self.jump_candidate_count = 0

        # 保留当前接口，记录外墙矩形切格参数。
        self.grid_base = {
            "cell_size_x": 0,
            "cell_size_y": 0,
            "grid_min_x": 0,
            "grid_min_y": 0,
        }

    def reset_coord_filter(self):
        """
        清空坐标滤波状态。

        当外墙矩形重新识别、地图坐标系变化，或小车被人工搬动时调用。
        """
        self.coord_filter_initialized = False
        self.filtered_car_x = 0.0
        self.filtered_car_y = 0.0
        self.last_raw_car_x = 0.0
        self.last_raw_car_y = 0.0
        self.jump_candidate_x = None
        self.jump_candidate_y = None
        self.jump_candidate_count = 0

    def filter_car_coord(self, raw_x, raw_y):
        """
        对视觉坐标进行极轻量处理：

            1. 单帧大跳变剔除，不计算平方根；
            2. 连续多帧在新位置附近时，确认这是真实位移；
            3. 对接受的坐标执行一阶 EMA。

        当前 ENABLE_COORD_FILTER=False 时直接返回原始视觉坐标，
        但完整滤波接口仍然保留。
        """
        if not ENABLE_COORD_FILTER:
            return (round(raw_x, 2), round(raw_y, 2))

        if not self.coord_filter_initialized:
            self.coord_filter_initialized = True
            self.filtered_car_x = raw_x
            self.filtered_car_y = raw_y
            self.last_raw_car_x = raw_x
            self.last_raw_car_y = raw_y
            self.jump_candidate_x = None
            self.jump_candidate_y = None
            self.jump_candidate_count = 0
            return (round(raw_x, 2), round(raw_y, 2))

        dx = raw_x - self.last_raw_car_x
        dy = raw_y - self.last_raw_car_y
        jump_sq = dx * dx + dy * dy

        accepted_x = None
        accepted_y = None

        if jump_sq <= COORD_MAX_JUMP_SQ:
            accepted_x = raw_x
            accepted_y = raw_y
            self.jump_candidate_x = None
            self.jump_candidate_y = None
            self.jump_candidate_count = 0
        else:
            if self.jump_candidate_x is None or self.jump_candidate_y is None:
                self.jump_candidate_x = raw_x
                self.jump_candidate_y = raw_y
                self.jump_candidate_count = 1
            else:
                candidate_dx = raw_x - self.jump_candidate_x
                candidate_dy = raw_y - self.jump_candidate_y
                candidate_dist_sq = (
                    candidate_dx * candidate_dx +
                    candidate_dy * candidate_dy
                )

                if candidate_dist_sq <= COORD_JUMP_CONFIRM_RADIUS_SQ:
                    self.jump_candidate_x = raw_x
                    self.jump_candidate_y = raw_y
                    self.jump_candidate_count += 1
                else:
                    self.jump_candidate_x = raw_x
                    self.jump_candidate_y = raw_y
                    self.jump_candidate_count = 1

            if self.jump_candidate_count >= COORD_JUMP_CONFIRM_FRAMES:
                accepted_x = raw_x
                accepted_y = raw_y
                self.jump_candidate_x = None
                self.jump_candidate_y = None
                self.jump_candidate_count = 0
            else:
                return (
                    round(self.filtered_car_x, 2),
                    round(self.filtered_car_y, 2)
                )

        self.filtered_car_x += COORD_EMA_ALPHA * (
            accepted_x - self.filtered_car_x
        )
        self.filtered_car_y += COORD_EMA_ALPHA * (
            accepted_y - self.filtered_car_y
        )

        self.last_raw_car_x = accepted_x
        self.last_raw_car_y = accepted_y

        return (
            round(self.filtered_car_x, 2),
            round(self.filtered_car_y, 2)
        )

    @staticmethod
    def detect_and_draw_wall_border(img, roi):
        """
        使用墙体颜色直接识别地图最外围矩形框。

        返回：
            [左上, 右上, 右下, 左下]

        注意：这里恢复的是旧版“外墙最小包围矩形”定位方式，
        但仍然使用当前版本的原始 snapshot 图像，不使用 lens_corr。
        """
        wall_blobs = img.find_blobs(
            [ELEMENT["wall"]["threshold"]],
            roi=roi,
            pixels_threshold=WALL_BORDER_PIXELS_THRESHOLD,
            area_threshold=WALL_BORDER_AREA_THRESHOLD,
            merge=True,
            margin=0
        )

        if not wall_blobs:
            return None

        min_x = WIDTH
        min_y = HEIGHT
        max_x = 0
        max_y = 0

        # 外墙和内部墙可能被分成多个 blob。
        # 对所有墙色块取最外侧 min/max，得到整张地图的矩形外框。
        for blob in wall_blobs:
            blob_left = blob.x()
            blob_top = blob.y()
            blob_right = blob.x() + blob.w()
            blob_bottom = blob.y() + blob.h()

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

        if wall_width < WALL_BORDER_MIN_WIDTH or \
                wall_height < WALL_BORDER_MIN_HEIGHT:
            return None

        # 向矩形内部收1像素，避免边界坐标刚好落在图像外或 blob 边缘。
        top_left = (min_x + 1, min_y + 1)
        top_right = (max_x - 1, min_y + 1)
        bottom_right = (max_x - 1, max_y - 1)
        bottom_left = (min_x + 1, max_y - 1)

        return [top_left, top_right, bottom_right, bottom_left]

    @staticmethod
    def get_car_roi_by_map_corners(map_corners):
        """
        根据外墙矩形四角生成小车识别 ROI。

        find_blobs() 使用矩形 ROI，因此这里直接使用外墙矩形范围。
        """
        if map_corners is None or len(map_corners) != 4:
            return None

        xs = [point[0] for point in map_corners]
        ys = [point[1] for point in map_corners]

        min_x = max(0, int(min(xs)))
        min_y = max(0, int(min(ys)))
        max_x = min(WIDTH, int(max(xs)) + 1)
        max_y = min(HEIGHT, int(max(ys)) + 1)

        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w <= 0 or roi_h <= 0:
            return None

        return (min_x, min_y, roi_w, roi_h)

    def set_map_corners(self, map_corners):
        """
        缓存新的外墙矩形坐标系、小车 ROI，并清空旧坐标滤波状态。
        """
        if map_corners is None:
            self.perspective_matrix = None
            self.car_roi = None
            return False

        self.perspective_matrix = self.compute_homography(map_corners)
        self.car_roi = self.get_car_roi_by_map_corners(map_corners)
        self.reset_coord_filter()

        return self.perspective_matrix is not None and self.car_roi is not None

    def detect_and_draw_car(self, img, map_corners):
        """
        根据外墙矩形坐标系检测小车坐标。

        外墙矩形表示完整16×12地图的外边界，因此坐标换算为：
            x = 归一化x × 16 - 0.5
            y = 归一化y × 12 - 0.5

        这样每个格子的中心仍然对应整数坐标：
            左上格中心为 (0, 0)，右下格中心为 (15, 11)。
        """
        if map_corners is None:
            return None

        if self.perspective_matrix is None:
            self.perspective_matrix = self.compute_homography(map_corners)

        if self.car_roi is None:
            self.car_roi = self.get_car_roi_by_map_corners(map_corners)

        if self.perspective_matrix is None or self.car_roi is None:
            return None

        head_blobs = img.find_blobs(
            [ELEMENT["car_head"]["threshold"]],
            roi=self.car_roi,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        back_blobs = img.find_blobs(
            [ELEMENT["car_back"]["threshold"]],
            roi=self.car_roi,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        if not head_blobs or not back_blobs:
            self.car_angle = None
            return None

        head_blob = max(head_blobs, key=lambda b: b.pixels())
        back_blob = max(back_blobs, key=lambda b: b.pixels())

        x1, y1 = head_blob.cx(), head_blob.cy()
        x2, y2 = back_blob.cx(), back_blob.cy()

        nx1, ny1 = self.apply_homography(
            self.perspective_matrix, x1, y1
        )
        nx2, ny2 = self.apply_homography(
            self.perspective_matrix, x2, y2
        )

        # 外墙矩形是完整16×12地图的外边界。
        ax1 = nx1 * GRID_X_NUM
        ay1 = ny1 * GRID_Y_NUM
        ax2 = nx2 * GRID_X_NUM
        ay2 = ny2 * GRID_Y_NUM

        ax = (ax1 + ax2) / 2 - 0.5
        ay = (ay1 + ay2) / 2 - 0.5

        # 明显越界则认为是误识别。
        if ax < -1 or ax > GRID_X_NUM or \
                ay < -1 or ay > GRID_Y_NUM:
            self.car_angle = None
            return None

        ax, ay = self.radial_distortion_correct(ax, ay)

        return self.filter_car_coord(ax, ay)

    def detect_grid_color_and_generate_map(
            self, img, map_corners, cell_inner_pad=None):
        """
        外墙矩形版地图识别。

        处理顺序：
            1. 外墙颜色识别得到完整地图矩形；
            2. 将矩形按像素边界划分为16列×12行；
            3. 最外围一圈不进行元素识别，直接固定为墙 '#';
            4. 只识别内部：
                   x = 1 ～ GRID_X_NUM-2，即1～14
                   y = 1 ～ GRID_Y_NUM-2，即1～10
            5. 每个内部格子向内缩 cell_inner_pad 像素；
            6. 最高元素占比低于 MIN_PIXEL_RATIO 时判为地面。

        用户所说的内部范围“(1,1)到(14,14)”在当前16×12地图中，
        y最大只能到10，因此这里按地图尺寸自动得到(1,1)到(14,10)。
        """
        if map_corners is None:
            return None

        if cell_inner_pad is None:
            cell_inner_pad = CELL_INNER_PAD

        # detect_and_draw_wall_border 返回轴对齐矩形四角。
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

        # 最外围一圈不做颜色判断，直接设为墙。
        for y_idx in range(GRID_Y_NUM):
            for x_idx in range(GRID_X_NUM):
                if x_idx == 0 or x_idx == GRID_X_NUM - 1 or \
                        y_idx == 0 or y_idx == GRID_Y_NUM - 1:
                    self.static_map[y_idx][x_idx] = \
                        ELEMENT["wall"]["symbol"]

        # 只识别内部格子：x=1..14，y=1..10。
        for y_idx in range(1, GRID_Y_NUM - 1):
            for x_idx in range(1, GRID_X_NUM - 1):
                # 使用边界公式，避免 int(cell_size) 造成累计误差。
                cell_x0 = int(min_x + x_idx * x_cell_size)
                cell_y0 = int(min_y + y_idx * y_cell_size)
                cell_x1 = int(min_x + (x_idx + 1) * x_cell_size)
                cell_y1 = int(min_y + (y_idx + 1) * y_cell_size)

                roi_x = cell_x0
                roi_y = cell_y0
                roi_w = cell_x1 - cell_x0
                roi_h = cell_y1 - cell_y0

                # 轻微缩小采样区域，减少格子边界和邻格颜色串入。
                if roi_w > 2 * cell_inner_pad and \
                        roi_h > 2 * cell_inner_pad:
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

                max_ratio = 0.0
                current_element = "ground"

                for element in ELEMENT_PRIORITY:
                    pixel_th = max(
                        5,
                        int(roi_total_pixels * ELEMENT_PIXEL_THRESHOLD_RATIO)
                    )

                    blobs = img.find_blobs(
                        [ELEMENT[element]["threshold"]],
                        roi=(roi_x, roi_y, roi_w, roi_h),
                        pixels_threshold=pixel_th,
                        merge=True,
                        margin=0
                    )

                    total_pixels = 0
                    for blob in blobs:
                        total_pixels += blob.pixels()

                    ratio = total_pixels / roi_total_pixels

                    if ratio > max_ratio:
                        max_ratio = ratio
                        current_element = element

                if max_ratio < MIN_PIXEL_RATIO:
                    current_element = "ground"

                self.static_map[y_idx][x_idx] = \
                    ELEMENT[current_element]["symbol"]

        return [row[:] for row in self.static_map]

    @staticmethod
    def compute_homography(corners):
        """
        根据四个矩形角点计算：图像像素坐标 -> 归一化坐标。
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
        应用透视矩阵，返回归一化坐标。
        """
        A, B, C, D, E, F, G, H_m, I_m = matrix

        w = G * x + H_m * y + I_m

        if w == 0:
            w = 0.0001

        x_norm = (A * x + B * y + C) / w
        y_norm = (D * x + E * y + F) / w

        return x_norm, y_norm

    @staticmethod
    def radial_distortion_correct(x, y):
        """
        四方向分段二次坐标矫正。
        保留当前版本参数和接口。
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
        print("".join(row))

    print("===================================\n")


def detect_wall_and_update_geometry(img):
    """
    识别外墙矩形并更新全局地图坐标系。
    """
    global map_corners

    detected_corners = gamesymbol.detect_and_draw_wall_border(img, ROI_WALL)

    if detected_corners is None:
        return False

    if not gamesymbol.set_map_corners(detected_corners):
        return False

    map_corners = detected_corners
    return True


def send_current_coord_once():
    """
    周期性坐标发送备用接口。
    优先使用已经缓存的外墙矩形；为空时重新识别一次。
    """
    global map_corners

    img = camera.capture()

    if map_corners is None:
        if not detect_wall_and_update_geometry(img):
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
# 摄像头与 LED 初始化
# ==========================
white = LED(4)
white.off()
camera = Camera()
WIDTH, HEIGHT = camera.getWidthHeight()


# ==========================
# 颜色阈值
# 保留当前最新版参数
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
# 外墙搜索区域：保留旧矩形框版本的有效范围。
ROI_WALL = (9, 10, 320, 230)

# 保留兼容常量；实际小车 ROI 会由外墙矩形自动生成。
ROI_CAR = (0, 0, 320, 240)

ELEMENT_PRIORITY = ["box", "goal", "bomb", "wall"]

GRID_X_NUM = 16
GRID_Y_NUM = 12

# 最高颜色占比低于该值时，内部格判定为地面。
MIN_PIXEL_RATIO = 0.5

# 每种元素的 blob 最低像素数为当前格采样面积的5%。
ELEMENT_PIXEL_THRESHOLD_RATIO = 0.05

# 每个内部格子的四边向内缩3像素，减少边界串色。
CELL_INNER_PAD = 3

DEBUG_PRINT_MAP = False

# ==========================
# 外墙矩形检测参数
# ==========================
WALL_BORDER_PIXELS_THRESHOLD = 300
WALL_BORDER_AREA_THRESHOLD = 300
WALL_BORDER_MIN_WIDTH = 20
WALL_BORDER_MIN_HEIGHT = 20


# ==========================
# 小车坐标轻量滤波参数
# ==========================
# 保持当前版本实际状态：关闭滤波，直接输出原始视觉坐标。
ENABLE_COORD_FILTER = False

COORD_EMA_ALPHA = 0.65

COORD_MAX_JUMP = 0.70
COORD_MAX_JUMP_SQ = COORD_MAX_JUMP * COORD_MAX_JUMP

COORD_JUMP_CONFIRM_FRAMES = 3

COORD_JUMP_CONFIRM_RADIUS = 0.70
COORD_JUMP_CONFIRM_RADIUS_SQ = (
    COORD_JUMP_CONFIRM_RADIUS * COORD_JUMP_CONFIRM_RADIUS
)


# ==========================
# 错误码说明
# ==========================
# 1：外墙矩形识别/初始化失败
# 2：map_corners 为空，没有可用地图外框
# 3：O 模式下小车识别失败
# 4：O 模式下地图识别失败
# 5：P 模式下单次坐标识别失败
# 6：O 模式下小车理论坐标越界，不能写入地图
ERR_WALL_INIT_FAIL = 1
ERR_NO_MAP_CORNERS = 2
ERR_CAR_DETECT_FAIL = 3
ERR_MAP_DETECT_FAIL = 4
ERR_COORD_DETECT_FAIL = 5
ERR_CAR_OUT_OF_RANGE = 6


# ==========================
# 周期性发送坐标参数
# 保留当前版本参数与接口；默认未启用。
# ==========================
COORD_SEND_INTERVAL_MS = 2000
auto_send_coord = False
last_coord_send_ms = 0


# ==========================
# 四方向分段二次坐标矫正参数
# 保留当前最新版参数
# ==========================
KX_LEFT = -0.02
KX_RIGHT = -0.02
KY_UP = 0.0
KY_DOWN = 0.0

DISTORT_CENTER_X = (GRID_X_NUM - 1) / 2
DISTORT_CENTER_Y = (GRID_Y_NUM - 1) / 2


# ==========================
# 主程序
# O：重新识别外墙矩形，返回初始坐标和地图
# P：返回当前坐标一次；尚无外框时先尝试识别外框
# S：停止备用周期发送状态
#
# 按用户上传的当前最新版行为：
# 普通循环不无条件发送坐标，只有 O/P 指令触发发送。
# ==========================
gamesymbol = GameSymbol()
map_corners = None

# time.sleep(0.1)

# img = camera.capture()

# if not detect_wall_and_update_geometry(img):
#     myuart.writeErr(ERR_WALL_INIT_FAIL)


# visual_ref = gamesymbol.detect_and_draw_car(img, map_corners)

# if visual_ref is None:
#     myuart.writeErr(ERR_CAR_DETECT_FAIL)


# myuart.writePmode(visual_ref[0], visual_ref[1])

# symbolmap = gamesymbol.detect_grid_color_and_generate_map(
#     img, map_corners
# )

# if symbolmap is None:
#     myuart.writeErr(ERR_MAP_DETECT_FAIL)

# theroitic_point = (
#     round(visual_ref[0]),
#     round(visual_ref[1])
# )

# if 0 <= theroitic_point[0] < GRID_X_NUM and \
#         0 <= theroitic_point[1] < GRID_Y_NUM:
#     symbolmap[theroitic_point[1]][theroitic_point[0]] = \
#         ELEMENT["car"]["symbol"]
# else:
#     myuart.writeErr(ERR_CAR_OUT_OF_RANGE)

# if DEBUG_PRINT_MAP:
#     print_symbol_map(symbolmap)


while True:
    img = camera.capture()
    cmd = myuart.readCmd()

    if cmd == b'O':
        # 每次收到 O 都重新识别外墙，避免沿用旧矩形位置。
        if not detect_wall_and_update_geometry(img):
            myuart.writeErr(ERR_WALL_INIT_FAIL)
            continue

        visual_ref = gamesymbol.detect_and_draw_car(img, map_corners)

        if visual_ref is None:
            myuart.writeErr(ERR_CAR_DETECT_FAIL)
            continue

        myuart.writePmode(visual_ref[0], visual_ref[1])

        symbolmap = gamesymbol.detect_grid_color_and_generate_map(
            img, map_corners
        )

        if symbolmap is None:
            myuart.writeErr(ERR_MAP_DETECT_FAIL)
            continue

        theroitic_point = (
            round(visual_ref[0]),
            round(visual_ref[1])
        )

        if 0 <= theroitic_point[0] < GRID_X_NUM and \
                0 <= theroitic_point[1] < GRID_Y_NUM:
            symbolmap[theroitic_point[1]][theroitic_point[0]] = \
                ELEMENT["car"]["symbol"]
        else:
            myuart.writeErr(ERR_CAR_OUT_OF_RANGE)
            continue

        if DEBUG_PRINT_MAP:
            print_symbol_map(symbolmap)

        myuart.writeMmode(symbolmap)

        # 保留当前版本状态：不自动开启周期发送。
        # auto_send_coord = True
        # last_coord_send_ms = time.ticks_ms()

        continue

    elif cmd == b'P':
        # 当前还没有外墙矩形时，先尝试建立地图坐标系。
        # if map_corners is None:
        #     if not detect_wall_and_update_geometry(img):
        #         myuart.writeErr(ERR_NO_MAP_CORNERS)
        #         continue
        co1,co2,co3,co4 = gamesymbol.detect_and_draw_wall_border(img,ROI_WALL)
        map_corners=[co1,co2,co3,co4]

        coord = gamesymbol.detect_and_draw_car(img, map_corners)

        if coord is None:
            myuart.writeErr(ERR_COORD_DETECT_FAIL)
            continue

        myuart.writePmode(coord[0], coord[1])
        continue

    elif cmd == b'S':
        auto_send_coord = False
        continue

    # 保留当前最新版的主循环行为：不做无条件连续坐标发送。
    # 若之后要恢复连续发送，可启用以下任一种逻辑。

    # # 方案一：每帧持续发送
    # if map_corners is not None:
    #     coord = gamesymbol.detect_and_draw_car(img, map_corners)
    #     if coord is not None:
    #         print(f"{coord[0]},{coord[1]}")
    #         # myuart.writePmode(coord[0], coord[1])

    # 方案二：按固定周期发送
    # if auto_send_coord:
    #     now_ms = time.ticks_ms()
    #     if time.ticks_diff(now_ms, last_coord_send_ms) >= \
    #             COORD_SEND_INTERVAL_MS:
    #         send_current_coord_once()
    #         last_coord_send_ms = now_ms
