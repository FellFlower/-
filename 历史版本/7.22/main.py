from pyb import LED
from machine import UART
import sensor, image, time

# clock = time.clock()

# ==========================
# 基础地图参数
# ==========================
GRID_X_NUM = 16
GRID_Y_NUM = 12
RECTIFIED_WIDTH = 320
RECTIFIED_HEIGHT = 240
GRID_CELL_SIZE_X = 20.0
GRID_CELL_SIZE_Y = 20.0
ROI_CAR = (0, 0, RECTIFIED_WIDTH, RECTIFIED_HEIGHT)

# ==========================
# AprilTag参数
# ==========================
TAG_FAMILY = image.TAG16H5
REQUIRED_TAG_IDS = (0, 1, 2, 3)
TAG_INIT_SAMPLE_COUNT = 3
TAG_INIT_MAX_RETRY = 0
TAG_INIT_FRAME_GAP_MS = 50

# Tag微调参数：
TAG_CORNER_OFFSETS = (
    (-8.0, -10.0),
    (8.0, -10.0),
    (8.0, 7.0),
    (-8.0, 7.0),
)

# ==========================
# 小车三帧加权坐标参数
# ==========================
CAR_COORD_WEIGHTS = (0.2, 0.3, 0.5)
CAR_COORD_FRAME_GAP_MS = 0

# ==========================
# 地图识别参数
# ==========================
ELEMENT_PRIORITY = ("box", "goal", "bomb", "wall")
MIN_PIXEL_RATIO = 0.5
CELL_SAMPLE_HALF = 0.4

# ==========================
# 错误码
# ==========================
ERR_TAG_INIT_FAIL = 1
ERR_NO_MAP_CORNERS = 2
ERR_CAR_DETECT_FAIL = 3
ERR_MAP_DETECT_FAIL = 4
ERR_COORD_DETECT_FAIL = 5
ERR_CAR_OUT_OF_RANGE = 6

# ==========================
# 周期发送参数
# ==========================
COORD_SEND_INTERVAL_MS = 2000
auto_send_coord = False
last_coord_send_ms = 0


class myUART:
    def __init__(self, uart_id, baudrate=115200):
        self.uart = UART(uart_id, baudrate=baudrate)

    def readFlag(self, byteFlag):
        assert byteFlag in (b'O', b'P', b'S'), \
            "we only have b'O', b'P' and b'S' 3 type flags"
        uart_num = self.uart.any()
        if uart_num == 0:
            return False
        uart_str = self.uart.read(uart_num)
        return bool(uart_str and byteFlag in uart_str)

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
        self.uart.write("&P%.2f,%.2f%%" % (num0, num1))

    def writeMmode(self, symbolmap):
        frame = "".join("".join(row) for row in symbolmap)
        self.uart.write("&M" + frame + "%")

    def writeErr(self, err_code):
        self.uart.write("&E" + str(err_code) + "%")

    def waitingSlave(self):
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

    def set_tag_mode(self):
        white.on()
        sensor.set_brightness(800)

    def set_normal_mode(self):
        white.off()
        sensor.set_brightness(200)

    def capture(self):
        return sensor.snapshot().lens_corr(0.35)

    @staticmethod
    def getWidthHeight():
        return sensor.width(), sensor.height()


class GameSymbol:
    def __init__(self):
        self.static_map = []

    def detect_apriltag_centers(self, img, draw=False):
        """
        单帧必须同时识别Tag 0、1、2、3才返回。
        返回顺序：[左上, 右上, 右下, 左下]。
        """
        if img is None:
            return None

        found = {}
        for tag in img.find_apriltags(families=TAG_FAMILY):
            tag_id = tag.id()
            if tag_id not in REQUIRED_TAG_IDS:
                continue

            rect = tag.rect()
            area = rect[2] * rect[3]

            if draw:
                img.draw_rectangle(rect, color=(255, 0, 0))
                img.draw_cross(tag.cx(), tag.cy(), color=(0, 255, 0))

            if tag_id not in found or area > found[tag_id][2]:
                found[tag_id] = (tag.cx(), tag.cy(), area)

        for tag_id in REQUIRED_TAG_IDS:
            if tag_id not in found:
                return None

        return [
            (found[0][0], found[0][1]),
            (found[1][0], found[1][1]),
            (found[2][0], found[2][1]),
            (found[3][0], found[3][1]),
        ]

    def apply_tag_corner_offsets(self, tag_centers):
        """
        将四个Tag平均中心按TAG_CORNER_OFFSETS扩展为地图外角。
        """
        if tag_centers is None:
            return None

        map_corners = []
        for tag_id in range(4):
            corner_x = int(round(
                tag_centers[tag_id][0] + TAG_CORNER_OFFSETS[tag_id][0]
            ))
            corner_y = int(round(
                tag_centers[tag_id][1] + TAG_CORNER_OFFSETS[tag_id][1]
            ))

            if corner_x < 0:
                corner_x = 0
            elif corner_x >= WIDTH:
                corner_x = WIDTH - 1

            if corner_y < 0:
                corner_y = 0
            elif corner_y >= HEIGHT:
                corner_y = HEIGHT - 1

            map_corners.append((corner_x, corner_y))

        return map_corners

    def init_apriltag_map(self, sample_count=3, max_retry=0):
        """
        收集sample_count张同时识别四个Tag的有效图，
        对各Tag中心取算术平均，再执行固定外扩微调。
        """
        camera.set_tag_mode()

        if sample_count <= 0:
            sample_count = 3

        sum_x = [0.0, 0.0, 0.0, 0.0]
        sum_y = [0.0, 0.0, 0.0, 0.0]
        valid_frame_count = 0
        capture_count = 0

        while valid_frame_count < sample_count:
            tag_centers = self.detect_apriltag_centers(
                camera.capture(), draw=False
            )
            capture_count += 1

            if tag_centers is not None:
                for tag_id in range(4):
                    sum_x[tag_id] += tag_centers[tag_id][0]
                    sum_y[tag_id] += tag_centers[tag_id][1]
                valid_frame_count += 1

                if TAG_INIT_FRAME_GAP_MS > 0:
                    time.sleep_ms(TAG_INIT_FRAME_GAP_MS)

            if (max_retry > 0 and capture_count >= max_retry
                    and valid_frame_count < sample_count):
                break

        camera.set_normal_mode()

        if valid_frame_count < sample_count:
            return None

        averaged_tag_centers = [
            (
                sum_x[tag_id] / valid_frame_count,
                sum_y[tag_id] / valid_frame_count
            )
            for tag_id in range(4)
        ]
        return self.apply_tag_corner_offsets(averaged_tag_centers)

    @staticmethod
    def rectify_image(img, map_corners):
        if img is None or map_corners is None:
            return None
        return img.rotation_corr(corners=map_corners)

    @staticmethod
    def pixel_to_map_coord(px, py):
        return px / GRID_CELL_SIZE_X, py / GRID_CELL_SIZE_Y

    @staticmethod
    def cell_center_to_pixel(x_idx, y_idx):
        return (
            (x_idx + 0.5) * GRID_CELL_SIZE_X,
            (y_idx + 0.5) * GRID_CELL_SIZE_Y
        )

    def detect_car_coord_once(self, rectified_img):
        """
        在一张拉正图中检测小车，返回未滤波的连续地图坐标。
        """
        if rectified_img is None:
            return None

        head_blobs = rectified_img.find_blobs(
            [ELEMENT["car_head"]["threshold"]],
            roi=ROI_CAR,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )
        back_blobs = rectified_img.find_blobs(
            [ELEMENT["car_back"]["threshold"]],
            roi=ROI_CAR,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        if not head_blobs or not back_blobs:
            return None

        head_blob = max(head_blobs, key=lambda blob: blob.pixels())
        back_blob = max(back_blobs, key=lambda blob: blob.pixels())

        car_px = (head_blob.cx() + back_blob.cx()) * 0.5
        car_py = (head_blob.cy() + back_blob.cy()) * 0.5
        raw_x, raw_y = self.pixel_to_map_coord(car_px, car_py)

        if (raw_x < -1.0 or raw_x > GRID_X_NUM + 1.0
                or raw_y < -1.0 or raw_y > GRID_Y_NUM + 1.0):
            return None

        return raw_x, raw_y

    def capture_weighted_car_coord(self, map_corners):
        """
        连续拍3张图并计算加权坐标：
            第1张×0.2 + 第2张×0.3 + 第3张×0.5

        返回：
            ((x, y), 第3张拉正图)

        任意一张小车识别失败时返回：
            (None, 最后一张已拍到的拉正图)
        """
        if map_corners is None:
            return None, None

        weighted_x = 0.0
        weighted_y = 0.0
        last_rectified_img = None

        for index in range(3):
            raw_img = camera.capture()
            last_rectified_img = self.rectify_image(raw_img, map_corners)
            coord = self.detect_car_coord_once(last_rectified_img)

            if coord is None:
                return None, last_rectified_img

            weight = CAR_COORD_WEIGHTS[index]
            weighted_x += coord[0] * weight
            weighted_y += coord[1] * weight

            if CAR_COORD_FRAME_GAP_MS > 0 and index < 2:
                time.sleep_ms(CAR_COORD_FRAME_GAP_MS)

        return (round(weighted_x, 2)-0.5, round(weighted_y, 2)-0.5), last_rectified_img

    def get_cell_roi(self, x_idx, y_idx):
        center_x, center_y = self.cell_center_to_pixel(x_idx, y_idx)
        half_w = CELL_SAMPLE_HALF * GRID_CELL_SIZE_X
        half_h = CELL_SAMPLE_HALF * GRID_CELL_SIZE_Y

        min_x = max(0, int(center_x - half_w))
        min_y = max(0, int(center_y - half_h))
        max_x = min(RECTIFIED_WIDTH, int(center_x + half_w))
        max_y = min(RECTIFIED_HEIGHT, int(center_y + half_h))

        roi_w = max_x - min_x
        roi_h = max_y - min_y
        if roi_w <= 0 or roi_h <= 0:
            return None
        return min_x, min_y, roi_w, roi_h

    def detect_grid_color_and_generate_map(self, rectified_img):
        """
        外围一圈固定为墙，只识别内部x=1~14、y=1~10。
        """
        if rectified_img is None:
            return None

        self.static_map = [
            [ELEMENT["ground"]["symbol"] for _ in range(GRID_X_NUM)]
            for _ in range(GRID_Y_NUM)
        ]

        for y_idx in range(GRID_Y_NUM):
            for x_idx in range(GRID_X_NUM):
                if (x_idx == 0 or x_idx == GRID_X_NUM - 1
                        or y_idx == 0 or y_idx == GRID_Y_NUM - 1):
                    self.static_map[y_idx][x_idx] = ELEMENT["wall"]["symbol"]

        for y_idx in range(1, GRID_Y_NUM - 1):
            for x_idx in range(1, GRID_X_NUM - 1):
                roi = self.get_cell_roi(x_idx, y_idx)
                if roi is None:
                    continue

                roi_x, roi_y, roi_w, roi_h = roi
                roi_total_pixels = roi_w * roi_h
                if roi_total_pixels <= 0:
                    continue

                element_ratio = {}
                pixel_threshold = max(5, int(roi_total_pixels * 0.05))

                for element in ELEMENT_PRIORITY:
                    blobs = rectified_img.find_blobs(
                        [ELEMENT[element]["threshold"]],
                        roi=(roi_x, roi_y, roi_w, roi_h),
                        pixels_threshold=pixel_threshold,
                        area_threshold=pixel_threshold,
                        merge=True,
                        margin=0
                    )
                    total_pixels = sum(blob.pixels() for blob in blobs)
                    element_ratio[element] = total_pixels / roi_total_pixels

                current_element = "ground"
                max_ratio = 0.0

                for element in ELEMENT_PRIORITY:
                    ratio = element_ratio[element]
                    if ratio > max_ratio:
                        max_ratio = ratio
                        current_element = element

                if max_ratio < MIN_PIXEL_RATIO:
                    current_element = "ground"

                self.static_map[y_idx][x_idx] = \
                    ELEMENT[current_element]["symbol"]

        return [row[:] for row in self.static_map]


# ==========================
# 串口、摄像头与LED初始化
# ==========================
myuart = myUART(12)
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


def capture_rectified_frame():
    if map_corners is None:
        return None
    return gamesymbol.rectify_image(camera.capture(), map_corners)


def send_current_coord_once():
    coord, _ = gamesymbol.capture_weighted_car_coord(map_corners)
    if coord is None:
        return False
    myuart.writePmode(coord[0], coord[1])
    return True


# ==========================
# AprilTag初始化
# ==========================
gamesymbol = GameSymbol()
map_corners = gamesymbol.init_apriltag_map(
    sample_count=TAG_INIT_SAMPLE_COUNT,
    max_retry=TAG_INIT_MAX_RETRY
)

if map_corners is None:
    myuart.writeErr(ERR_TAG_INIT_FAIL)


# ==========================
# 主程序
# O：三帧加权坐标 + 使用第三帧识别地图
# P：三帧加权坐标
# S：关闭备用周期发送
# ==========================
while True:
    img = camera.capture().rotation_corr(corners = map_corners)
    cmd = myuart.readCmd()

    if cmd == b'O':
        if map_corners is None:
            myuart.writeErr(ERR_NO_MAP_CORNERS)
            continue

        visual_ref, rectified_img = \
            gamesymbol.capture_weighted_car_coord(map_corners)

        if visual_ref is None:
            myuart.writeErr(ERR_CAR_DETECT_FAIL)
            continue

        myuart.writePmode(visual_ref[0], visual_ref[1])

        symbolmap = gamesymbol.detect_grid_color_and_generate_map(
            rectified_img
        )
        if symbolmap is None:
            myuart.writeErr(ERR_MAP_DETECT_FAIL)
            continue

        car_grid_x = int(visual_ref[0])
        car_grid_y = int(visual_ref[1])

        if (0 <= car_grid_x < GRID_X_NUM
                and 0 <= car_grid_y < GRID_Y_NUM):
            symbolmap[car_grid_y][car_grid_x] = ELEMENT["car"]["symbol"]
        else:
            myuart.writeErr(ERR_CAR_OUT_OF_RANGE)
            continue

        myuart.writeMmode(symbolmap)
        continue

    if cmd == b'P':
        if map_corners is None:
            myuart.writeErr(ERR_NO_MAP_CORNERS)
            continue

        coord, _ = gamesymbol.capture_weighted_car_coord(map_corners)
        if coord is None:
            myuart.writeErr(ERR_COORD_DETECT_FAIL)
            continue

        myuart.writePmode(coord[0], coord[1])
        continue

    if cmd == b'S':
        auto_send_coord = False
        continue

    if map_corners is None:
        time.sleep_ms(10)
        continue

    # print(clock.fps())

    # coord = gamesymbol.capture_weighted_car_coord(map_corners)
    # if coord is None:
    #     continue
    # print(f"{coord[0]},{coord[1]}")


    # 备用周期发送：
    # if auto_send_coord:
    #     now_ms = time.ticks_ms()
    #     if time.ticks_diff(now_ms, last_coord_send_ms) >= \
    #             COORD_SEND_INTERVAL_MS:
    #         send_current_coord_once()
    #         last_coord_send_ms = now_ms
