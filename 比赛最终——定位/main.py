from pyb import LED
from machine import UART
import sensor, image, time


# ==========================
# 基础地图参数
# ==========================
GRID_X_NUM = 16
GRID_Y_NUM = 12

RECTIFIED_WIDTH = 320
RECTIFIED_HEIGHT = 240

GRID_CELL_SIZE_X = 20.0
GRID_CELL_SIZE_Y = 20.0

# 拉正后的整张地图区域
ROI_CAR = (0, 0, RECTIFIED_WIDTH, RECTIFIED_HEIGHT)


# ==========================
# 摄像头参数
# ==========================
CAMERA_FPS = 60


# ==========================
# AprilTag参数
# ==========================
TAG_FAMILY = image.TAG16H5
REQUIRED_TAG_IDS = (0, 1, 2, 3)

TAG_INIT_SAMPLE_COUNT = 3
TAG_INIT_MAX_RETRY = 0
TAG_INIT_FRAME_GAP_MS = 50

# Tag微调参数
TAG_CORNER_OFFSETS = (
    (-9, -10.0),
    (10.0, -10.0),
    (10.0, 8.0),
    (-9, 8.0),
)


# ==========================
# 周期性AprilTag重新标定参数
# ==========================
# True：开启周期自动重新标定
# False：关闭周期自动重新标定，只保留上电初始化和串口M手动重标定
ENABLE_AUTO_RECALIBRATE = True

# 两次“标定完成”之间正常发送坐标的时间，单位ms
# 例如15000表示每正常运行15秒重新标定一次
AUTO_RECALIBRATE_INTERVAL_MS = 40000

# 自动/手动重新标定时需要采集的有效Tag帧数
RECALIBRATE_SAMPLE_COUNT = 5

# 重新标定最多尝试多少帧
# 必须大于0，防止周期标定时Tag识别失败导致程序永久卡住
RECALIBRATE_MAX_RETRY = 30


# ==========================
# 小车动态ROI参数
# ==========================
# 第一帧全图搜索；
# 之后以上一帧小车中心为中心，只搜索80×80；
# 如果局部ROI识别失败，同一帧自动退回全图重新捕获。
CAR_TRACKING_ROI_SIZE = 80


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
# 连续坐标发送开关
# ==========================
# True：
#   主循环没有收到O/P/S/B命令时，持续单帧定位并发送P坐标。
# False：
#   不自动发送，只响应命令。
auto_send_coord = True


# ==========================
# 坐标EMA滤波参数
# ==========================
# True：开启EMA，所有最终小车坐标统一经过EMA后再输出。
# False：关闭EMA，行为与原代码一致，直接输出单帧原始坐标。
ENABLE_COORD_EMA = False

# EMA系数：越大越跟手，越小越平滑。
COORD_EMA_ALPHA = 0.65


class myUART:
    def __init__(self, uart_id, baudrate=115200):
        self.uart = UART(uart_id, baudrate=baudrate)

    def readFlag(self, byteFlag):
        assert byteFlag in (b'O', b'P', b'S', b'B', b'M'), \
            "we only have b'O', b'P', b'S', b'B', b'M' 5 type flags"

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

        if b'B' in uart_str:
            return b'B'

        if b'T' in uart_str:
            return b'T'

        if b'F' in uart_str:
            return b'F'

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

    def sendEmpty(self):
        self.uart.write("&K%")


class Camera:
    def __init__(self):
        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)

        # 关键修改：
        # 将摄像头目标帧率设为60FPS。
        sensor.set_framerate(CAMERA_FPS)

        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)
        sensor.set_auto_exposure(False)
        sensor.set_brightness(200)

    def set_tag_mode(self):
        white.on()
        sensor.set_brightness(1000)

        # AprilTag初始化属于低频流程，
        # 这里保留原来的稳定等待时间。
        sensor.skip_frames(time=300)

    def set_normal_mode(self):
        white.off()
        sensor.set_brightness(200)

        # 从Tag模式切回正常颜色识别后等待画面稳定。
        sensor.skip_frames(time=300)

    def capture(self):
        return sensor.snapshot()

    @staticmethod
    def getWidthHeight():
        return sensor.width(), sensor.height()


class GameSymbol:
    def __init__(self):
        self.static_map = []

        # 上一帧成功识别的小车中心。
        # 注意：这里保存的是“拉正图”的像素坐标，
        # 因为小车识别发生在rotation_corr之后。
        self.last_car_px = None
        self.last_car_py = None

        # 坐标EMA状态
        self.coord_ema_initialized = False
        self.ema_car_x = 0.0
        self.ema_car_y = 0.0

    def reset_car_tracking(self):
        """
        清空局部ROI跟踪状态。
        下一次坐标检测会重新使用整张地图ROI捕获小车。
        """
        self.last_car_px = None
        self.last_car_py = None

    def reset_coord_ema(self):
        """清空EMA状态；地图坐标系重新初始化时调用。"""
        self.coord_ema_initialized = False
        self.ema_car_x = 0.0
        self.ema_car_y = 0.0

    def apply_coord_ema(self, coord):
        """对连续坐标执行EMA；关闭开关时原样返回。"""
        if coord is None:
            return None
        raw_x, raw_y = coord
        if not ENABLE_COORD_EMA:
            return raw_x, raw_y
        if not self.coord_ema_initialized:
            self.coord_ema_initialized = True
            self.ema_car_x = raw_x
            self.ema_car_y = raw_y
            return raw_x, raw_y
        self.ema_car_x += COORD_EMA_ALPHA * (raw_x - self.ema_car_x)
        self.ema_car_y += COORD_EMA_ALPHA * (raw_y - self.ema_car_y)
        return self.ema_car_x, self.ema_car_y

    def detect_apriltag_centers(self, img, draw=False):
        """
        单帧必须同时识别Tag 0、1、2、3才返回。

        返回顺序：
            [左上, 右上, 右下, 左下]
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
                found[tag_id] = (
                    tag.cx(),
                    tag.cy(),
                    area
                )

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
                tag_centers[tag_id][0]
                + TAG_CORNER_OFFSETS[tag_id][0]
            ))

            corner_y = int(round(
                tag_centers[tag_id][1]
                + TAG_CORNER_OFFSETS[tag_id][1]
            ))

            if corner_x < 0:
                corner_x = 0
            elif corner_x >= WIDTH:
                corner_x = WIDTH - 1

            if corner_y < 0:
                corner_y = 0
            elif corner_y >= HEIGHT:
                corner_y = HEIGHT - 1

            map_corners.append(
                (corner_x, corner_y)
            )

        return map_corners

    def init_apriltag_map(self, sample_count=5, max_retry=0):
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
                camera.capture(),
                draw=False
            )

            capture_count += 1

            if tag_centers is not None:
                for tag_id in range(4):
                    sum_x[tag_id] += tag_centers[tag_id][0]
                    sum_y[tag_id] += tag_centers[tag_id][1]

                valid_frame_count += 1

                if TAG_INIT_FRAME_GAP_MS > 0:
                    time.sleep_ms(TAG_INIT_FRAME_GAP_MS)

            if (max_retry > 0
                    and capture_count >= max_retry
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

        # 地图几何重新初始化后，清除旧的小车ROI和EMA状态。
        self.reset_car_tracking()
        self.reset_coord_ema()

        return self.apply_tag_corner_offsets(
            averaged_tag_centers
        )

    @staticmethod
    def rectify_image(img, map_corners):
        """
        将原始摄像头图像拉正到320×240地图坐标系。
        """
        if img is None or map_corners is None:
            return None

        return img.rotation_corr(
            corners=map_corners
        )

    @staticmethod
    def pixel_to_map_coord(px, py):
        """
        拉正图像素 -> 连续地图坐标。

        由于每格20×20像素：
            px / 20
            py / 20

        之后输出坐标时再统一减0.5，
        使格子中心对应整数游戏坐标。
        """
        return (
            px / GRID_CELL_SIZE_X,
            py / GRID_CELL_SIZE_Y
        )

    @staticmethod
    def cell_center_to_pixel(x_idx, y_idx):
        return (
            (x_idx + 0.5) * GRID_CELL_SIZE_X,
            (y_idx + 0.5) * GRID_CELL_SIZE_Y
        )

    @staticmethod
    def get_tracking_roi(center_x, center_y,
                         size=CAR_TRACKING_ROI_SIZE):
        """
        以上一帧成功识别的小车中心为中心，
        在拉正图上生成size×size局部搜索ROI。

        默认：
            80×80

        靠近边缘时，不把ROI尺寸截小，
        而是整体向图像内部平移，
        尽量始终保持80×80。
        """
        roi_w = min(size, RECTIFIED_WIDTH)
        roi_h = min(size, RECTIFIED_HEIGHT)

        center_x = int(round(center_x))
        center_y = int(round(center_y))

        roi_x = center_x - roi_w // 2
        roi_y = center_y - roi_h // 2

        max_x = RECTIFIED_WIDTH - roi_w
        max_y = RECTIFIED_HEIGHT - roi_h

        if roi_x < 0:
            roi_x = 0
        elif roi_x > max_x:
            roi_x = max_x

        if roi_y < 0:
            roi_y = 0
        elif roi_y > max_y:
            roi_y = max_y

        return (
            int(roi_x),
            int(roi_y),
            int(roi_w),
            int(roi_h)
        )

    def detect_car_coord_once(self,
                              rectified_img,
                              search_roi=None):
        """
        在一张已经拉正的图中检测一次小车。

        search_roi:
            None    -> 搜索整张地图ROI
            非None  -> 只搜索指定局部ROI

        返回：
            (raw_x, raw_y)

        这里返回的是减0.5之前的连续地图坐标。
        """
        if rectified_img is None:
            return None

        if search_roi is None:
            current_roi = ROI_CAR
        else:
            current_roi = search_roi

        head_blobs = rectified_img.find_blobs(
            [ELEMENT["car_head"]["threshold"]],
            roi=current_roi,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        back_blobs = rectified_img.find_blobs(
            [ELEMENT["car_back"]["threshold"]],
            roi=current_roi,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
            margin=0
        )

        if not head_blobs or not back_blobs:
            return None

        head_blob = max(
            head_blobs,
            key=lambda blob: blob.pixels()
        )

        back_blob = max(
            back_blobs,
            key=lambda blob: blob.pixels()
        )

        # 小车像素中心 = 车头、车尾色块中心的中点
        car_px = (
            head_blob.cx() + back_blob.cx()
        ) * 0.5

        car_py = (
            head_blob.cy() + back_blob.cy()
        ) * 0.5

        raw_x, raw_y = self.pixel_to_map_coord(
            car_px,
            car_py
        )

        # 明显越界则认为本次识别错误
        if (raw_x < -1.0
                or raw_x > GRID_X_NUM + 1.0
                or raw_y < -1.0
                or raw_y > GRID_Y_NUM + 1.0):
            return None

        # 只有识别成功后才更新下一帧ROI中心
        self.last_car_px = car_px
        self.last_car_py = car_py

        return raw_x, raw_y

    def detect_car_coord_tracking_once(self,
                                       rectified_img):
        """
        动态ROI小车检测。

        流程：
            1. 没有历史位置：
               整张地图ROI搜索。

            2. 已有上一帧位置：
               以上一帧车中心为中心，
               搜索80×80局部ROI。

            3. 局部ROI失败：
               在同一张图上立刻退回整张地图ROI重新捕获。

        成功返回：
            (raw_x, raw_y)

        失败返回：
            None
        """
        if rectified_img is None:
            return None

        # 第一帧 / 跟踪状态被清空
        if (self.last_car_px is None
                or self.last_car_py is None):

            return self.detect_car_coord_once(
                rectified_img,
                search_roi=None
            )

        # 生成80×80局部ROI
        tracking_roi = self.get_tracking_roi(
            self.last_car_px,
            self.last_car_py,
            size=CAR_TRACKING_ROI_SIZE
        )

        # 先局部搜索
        coord = self.detect_car_coord_once(
            rectified_img,
            search_roi=tracking_roi
        )

        if coord is not None:
            return coord

        # 局部搜索失败：
        # 清空旧位置，再在同一帧上执行全图回捕。
        self.reset_car_tracking()

        return self.detect_car_coord_once(
            rectified_img,
            search_roi=None
        )

    @staticmethod
    def format_car_coord(raw_coord):
        """
        将内部连续坐标转成最终对外发送坐标。

        原有坐标定义保持不变：
            output = raw - 0.5
        """
        if raw_coord is None:
            return None

        return (
            round(raw_coord[0] - 0.5, 2),
            round(raw_coord[1] - 0.5, 2)
        )

    def detect_output_car_coord(self,
                                rectified_img,
                                use_tracking=True):
        """
        在已经拉正的一张图上获取最终输出坐标。
        不拍第二张图，不做多帧加权。
        """
        if use_tracking:
            raw_coord = \
                self.detect_car_coord_tracking_once(
                    rectified_img
                )
        else:
            raw_coord = self.detect_car_coord_once(
                rectified_img,
                search_roi=None
            )

        filtered_coord = self.apply_coord_ema(raw_coord)

        return self.format_car_coord(
            filtered_coord
        )

    def capture_single_car_coord(self,
                                 map_corners,
                                 use_tracking=True):
        """
        单帧、不加权的小车坐标链路：

            camera.capture()
                ↓
            rotation_corr()
                ↓
            80×80动态ROI检测
                ↓
            坐标输出

        返回：
            ((x, y), rectified_img)

        失败：
            (None, rectified_img)
        """
        if map_corners is None:
            return None, None

        raw_img = camera.capture()

        rectified_img = self.rectify_image(
            raw_img,
            map_corners
        )

        if rectified_img is None:
            return None, None

        coord = self.detect_output_car_coord(
            rectified_img,
            use_tracking=use_tracking
        )

        return coord, rectified_img

    def get_cell_roi(self, x_idx, y_idx):
        center_x, center_y = \
            self.cell_center_to_pixel(
                x_idx,
                y_idx
            )

        half_w = (
            CELL_SAMPLE_HALF
            * GRID_CELL_SIZE_X
        )

        half_h = (
            CELL_SAMPLE_HALF
            * GRID_CELL_SIZE_Y
        )

        min_x = max(
            0,
            int(center_x - half_w)
        )

        min_y = max(
            0,
            int(center_y - half_h)
        )

        max_x = min(
            RECTIFIED_WIDTH,
            int(center_x + half_w)
        )

        max_y = min(
            RECTIFIED_HEIGHT,
            int(center_y + half_h)
        )

        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w <= 0 or roi_h <= 0:
            return None

        return (
            min_x,
            min_y,
            roi_w,
            roi_h
        )

    def detect_grid_color_and_generate_map(
            self,
            rectified_img):
        """
        外围一圈固定为墙，
        只识别内部：
            x = 1 ~ 14
            y = 1 ~ 10
        """
        if rectified_img is None:
            return None

        self.static_map = [
            [
                ELEMENT["ground"]["symbol"]
                for _ in range(GRID_X_NUM)
            ]
            for _ in range(GRID_Y_NUM)
        ]

        # 外圈直接设置为墙
        for y_idx in range(GRID_Y_NUM):
            for x_idx in range(GRID_X_NUM):
                if (x_idx == 0
                        or x_idx == GRID_X_NUM - 1
                        or y_idx == 0
                        or y_idx == GRID_Y_NUM - 1):

                    self.static_map[y_idx][x_idx] = \
                        ELEMENT["wall"]["symbol"]

        # 内部格子颜色识别
        for y_idx in range(
                1,
                GRID_Y_NUM - 1):

            for x_idx in range(
                    1,
                    GRID_X_NUM - 1):

                roi = self.get_cell_roi(
                    x_idx,
                    y_idx
                )

                if roi is None:
                    continue

                roi_x, roi_y, roi_w, roi_h = roi

                roi_total_pixels = (
                    roi_w * roi_h
                )

                if roi_total_pixels <= 0:
                    continue

                element_ratio = {}

                pixel_threshold = max(
                    5,
                    int(
                        roi_total_pixels
                        * 0.05
                    )
                )

                for element in ELEMENT_PRIORITY:

                    blobs = rectified_img.find_blobs(
                        [ELEMENT[element]["threshold"]],
                        roi=(
                            roi_x,
                            roi_y,
                            roi_w,
                            roi_h
                        ),
                        pixels_threshold=pixel_threshold,
                        area_threshold=pixel_threshold,
                        merge=True,
                        margin=0
                    )

                    total_pixels = sum(
                        blob.pixels()
                        for blob in blobs
                    )

                    element_ratio[element] = (
                        total_pixels
                        / roi_total_pixels
                    )

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

        return [
            row[:]
            for row in self.static_map
        ]


# ==========================
# 串口、摄像头与LED初始化
# ==========================
myuart = myUART(12)

white = LED(4)

camera = Camera()

WIDTH, HEIGHT = \
    camera.getWidthHeight()


# ==========================
# 颜色阈值
# ==========================
ELEMENT = {
    "wall": {
        "threshold": (
            30, 89,
            -7, 47,
            -79, 7
        ),
        "symbol": "#"
    },

    "box": {
        "threshold": (
            17, 95,
            -42, 34,
            21, 93
        ),
        "symbol": "$"
    },

    "car_head": {
        "threshold": (
            69, 96,
            -63, -15,
            -56, 1
        ),
        "symbol": None
    },

    "car_back": {
        "threshold": (
            64, 91,
            -92, -52,
            11, 98
        ),
        "symbol": None
    },

    "goal": {
        "threshold": (
            17, 88,
            64, 107,
            -86, -29
        ),
        "symbol": "."
    },

    "bomb": {
        "threshold": (
            24, 77,
            24, 95,
            -27, 86
        ),
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
    """
    只拍一帧并拉正。
    """
    if map_corners is None:
        return None

    raw_img = camera.capture()

    return gamesymbol.rectify_image(
        raw_img,
        map_corners
    )


def send_current_coord_once():
    """
    连续定位：
        单帧
        + 80×80动态ROI
        + 不加权
        + 立即UART发送
    """
    if map_corners is None:
        return False

    coord, _ = \
        gamesymbol.capture_single_car_coord(
            map_corners,
            use_tracking=True
        )

    if coord is None:
        return False

    myuart.writePmode(
        coord[0],
        coord[1]
    )

    return True


def recalibrate_map_corners():
    """
    运行中重新进行AprilTag地图标定。

    标定期间本函数阻塞执行，因此主循环不会发送P坐标。
    标定成功：
        使用新的map_corners。
    标定失败：
        保留旧的map_corners。
    无论成功失败：
        切回正常模式后立即返回主循环，并从此刻重新计算周期。
    """
    global map_corners, last_recalibrate_ms

    old_map_corners = map_corners

    new_map_corners = gamesymbol.init_apriltag_map(
        sample_count=RECALIBRATE_SAMPLE_COUNT,
        max_retry=RECALIBRATE_MAX_RETRY
    )

    if new_map_corners is not None:
        map_corners = new_map_corners
        success = True
        print("Map recalibrate OK:", map_corners)
    else:
        # 本次Tag识别失败时不破坏原来的坐标系
        map_corners = old_map_corners
        success = False
        print("Map recalibrate FAIL, keep old map_corners")

    # 从重新标定结束的这一刻开始重新计时。
    # 这样保证两次标定之间有完整的正常坐标发送时间。
    last_recalibrate_ms = time.ticks_ms()

    return success


def send_map_from_frame(visual_ref,
                        rectified_img):
    """
    使用已经拍到的同一张拉正图识别地图并发送M帧。

    成功：
        返回True

    失败：
        自动发送对应错误码并返回False
    """
    if visual_ref is None:
        myuart.writeErr(
            ERR_CAR_DETECT_FAIL
        )
        return False

    symbolmap = \
        gamesymbol.detect_grid_color_and_generate_map(
            rectified_img
        )

    if symbolmap is None:
        myuart.writeErr(
            ERR_MAP_DETECT_FAIL
        )
        return False

    car_grid_x = int(
        visual_ref[0] + 0.5
    )

    car_grid_y = int(
        visual_ref[1] + 0.5
    )

    if (0 <= car_grid_x < GRID_X_NUM
            and 0 <= car_grid_y < GRID_Y_NUM):

        symbolmap[car_grid_y][car_grid_x] = \
            ELEMENT["car"]["symbol"]

    else:
        myuart.writeErr(
            ERR_CAR_OUT_OF_RANGE
        )
        return False

    myuart.writeMmode(
        symbolmap
    )

    return True


# ==========================
# AprilTag初始化
# ==========================
gamesymbol = GameSymbol()

map_corners = \
    gamesymbol.init_apriltag_map(
        sample_count=TAG_INIT_SAMPLE_COUNT,
        max_retry=TAG_INIT_MAX_RETRY
    )

if map_corners is None:
    myuart.writeErr(
        ERR_TAG_INIT_FAIL
    )

# 周期重新标定从上电初始化完成后开始计时
last_recalibrate_ms = time.ticks_ms()


# ==========================
# 主程序
#
# O：
#   单帧小车坐标
#   + 使用同一帧识别地图
#
# P：
#   单帧小车坐标
#
# B：
#   单帧检查box
#   + 若存在，使用同一帧检测小车和地图
# S：
#   关闭默认连续坐标发送
#
# M：
#   立即重新识别四个AprilTag并更新map_corners
#
# 自动重新标定：
#   达到AUTO_RECALIBRATE_INTERVAL_MS后暂停P坐标发送，
#   完成Tag标定后立即恢复连续坐标发送。
# ==========================

stop_tag = 0
while True:
    # --------------------------
    # 周期自动重新标定
    # --------------------------
    # 放在读取串口命令之前，避免已经读到的命令在标定时被丢掉。
    # 仅在连续坐标发送开启时执行自动标定。
    if (ENABLE_AUTO_RECALIBRATE
            and auto_send_coord
            and map_corners is not None):

        now_ms = time.ticks_ms()

        if time.ticks_diff(
                now_ms,
                last_recalibrate_ms
        ) >= AUTO_RECALIBRATE_INTERVAL_MS:

            # recalibrate_map_corners()阻塞期间不会执行任何P坐标发送。
            # 函数返回后continue进入下一轮，立即恢复正常坐标发送。
            recalibrate_map_corners()
            time.sleep_ms(300)
            continue

    cmd = myuart.readCmd()

    # --------------------------
    # T模式：手动重新标定
    # --------------------------
    # if cmd == b'T':
    #     # 标定期间不会发送P坐标。
    #     # 成功则换用新map_corners，失败则保留旧map_corners。
    #     recalibrate_map_corners()
    #     continue

    # --------------------------
    # O模式
    # --------------------------
    if cmd == b'O':

        if map_corners is None:
            myuart.writeErr(
                ERR_NO_MAP_CORNERS
            )
            continue

        # 只拍一帧
        visual_ref, rectified_img = \
            gamesymbol.capture_single_car_coord(
                map_corners,
                use_tracking=True
            )

        if visual_ref is None:
            myuart.writeErr(
                ERR_CAR_DETECT_FAIL
            )
            continue

        # 先发送当前位置
        myuart.writePmode(
            visual_ref[0],
            visual_ref[1]
        )

        # 再使用同一张图识别地图
        send_map_from_frame(
            visual_ref,
            rectified_img
        )

        continue

    # --------------------------
    # P模式
    # --------------------------
    if cmd == b'P':

        if map_corners is None:
            myuart.writeErr(
                ERR_NO_MAP_CORNERS
            )
            continue

        coord, _ = \
            gamesymbol.capture_single_car_coord(
                map_corners,
                use_tracking=True
            )

        if coord is None:
            myuart.writeErr(
                ERR_COORD_DETECT_FAIL
            )
            continue

        myuart.writePmode(
            coord[0],
            coord[1]
        )

        continue

    # --------------------------
    # S模式
    # --------------------------
    if cmd == b'S':
        auto_send_coord = False
        continue

    # --------------------------
    # B模式
    # --------------------------
    if cmd == b'B':

        if map_corners is None:
            myuart.writeErr(
                ERR_NO_MAP_CORNERS
            )
            continue

        # B模式也只拍一帧
        rectified_img = \
            capture_rectified_frame()

        if rectified_img is None:
            myuart.writeErr(
                ERR_MAP_DETECT_FAIL
            )
            continue

        box_blobs = \
            rectified_img.find_blobs(
                [ELEMENT["box"]["threshold"]],
                pixels_threshold=100,
                area_threshold=100,
                merge=True,
                margin=0
            )

        if not box_blobs:
            myuart.sendEmpty()
            continue

        # 在刚才那一张图上直接检测小车，
        # 不再额外capture，不使用三帧加权。
        visual_ref = \
            gamesymbol.detect_output_car_coord(
                rectified_img,
                use_tracking=True
            )

        if visual_ref is None:
            myuart.writeErr(
                ERR_CAR_DETECT_FAIL
            )
            continue

        myuart.writePmode(
            visual_ref[0],
            visual_ref[1]
        )

        send_map_from_frame(
            visual_ref,
            rectified_img
        )

        continue

    # --------------------------
    # 默认连续坐标发送
    # --------------------------
    if cmd == b'F':
        stop_tag = 1
        continue

    # coord, _ = \
    #     gamesymbol.capture_single_car_coord(
    #         map_corners,
    #         use_tracking=True
    #     )
    # if coord != None:
    #     print(f"{coord[0]},{coord[1]}")

    # if map_corners is None:
    #     time.sleep_ms(10)
    #     continue

    if auto_send_coord and not stop_tag:
        # 不再人为限制35ms。
        # 每处理完一帧立即开始下一帧，
        # 让sensor.set_framerate(60)决定实际节拍。
        send_current_coord_once()
    else:
        # 自动发送关闭后避免空转占满CPU。
        time.sleep_ms(1)


