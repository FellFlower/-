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
        # 坐标固定保留两位小数，避免出现 9.510001 这类浮点尾数。
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

        # # 先让相机稳定一段时间，然后固定曝光。
        # sensor.skip_frames(time=500)
        # sensor.set_auto_exposure(False, exposure_us=100)

        sensor.set_brightness(200)
        # sensor.skip_frames(time=200)

    def set_tag_mode(self):
        white.on()
        sensor.set_brightness(800)
        # sensor.skip_frames(time=200)

    def set_normal_mode(self):
        white.off()
        sensor.set_brightness(200)
        # sensor.skip_frames(time=200)

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
        self.car_roi = None
        self.static_map = []

        # ==========================
        # 小车坐标轻量滤波状态
        # ==========================
        # filtered_car_x/y：当前发送给下位机的稳定坐标
        # last_raw_car_x/y：上一帧被接受的原始视觉坐标
        # jump_candidate_*：大跳变候选，用于连续多帧确认真实位移
        self.coord_filter_initialized = False
        self.filtered_car_x = 0.0
        self.filtered_car_y = 0.0
        self.last_raw_car_x = 0.0
        self.last_raw_car_y = 0.0
        self.jump_candidate_x = None
        self.jump_candidate_y = None
        self.jump_candidate_count = 0

        self.grid_base = {
            "cell_size_x": 0,
            "cell_size_y": 0,
            "grid_min_x": 0,
            "grid_min_y": 0,
        }

    def reset_coord_filter(self):
        """
        清空坐标滤波状态。

        一般不需要主动调用；只有在地图坐标系重新初始化、
        或者小车被人工搬到很远的位置时才需要重置。
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

        大跳变尚未确认时，继续返回上一稳定坐标，
        因而不会破坏主循环“持续发送坐标”的现有用法。
        """
        if not ENABLE_COORD_FILTER:
            return (round(raw_x, 2), round(raw_y, 2))

        # 第一帧直接作为滤波初始值，避免启动时从 (0, 0) 缓慢追踪。
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
            # 正常移动：立即接受，并清除尚未确认的大跳变候选。
            accepted_x = raw_x
            accepted_y = raw_y
            self.jump_candidate_x = None
            self.jump_candidate_y = None
            self.jump_candidate_count = 0
        else:
            # 当前坐标相对上一有效原始坐标跳变过大。
            # 不立刻采用，先检查后续帧是否持续出现在相近的新位置。
            if self.jump_candidate_x is None or self.jump_candidate_y is None:
                self.jump_candidate_x = raw_x
                self.jump_candidate_y = raw_y
                self.jump_candidate_count = 1
            else:
                candidate_dx = raw_x - self.jump_candidate_x
                candidate_dy = raw_y - self.jump_candidate_y
                candidate_dist_sq = (candidate_dx * candidate_dx +
                                     candidate_dy * candidate_dy)

                if candidate_dist_sq <= COORD_JUMP_CONFIRM_RADIUS_SQ:
                    # 新的异常帧和上一候选相近，增加确认计数。
                    self.jump_candidate_x = raw_x
                    self.jump_candidate_y = raw_y
                    self.jump_candidate_count += 1
                else:
                    # 候选位置本身也在乱跳，从当前帧重新计数。
                    self.jump_candidate_x = raw_x
                    self.jump_candidate_y = raw_y
                    self.jump_candidate_count = 1

            if self.jump_candidate_count >= COORD_JUMP_CONFIRM_FRAMES:
                # 连续多帧证明确实到了新位置，允许滤波器跟过去。
                accepted_x = raw_x
                accepted_y = raw_y
                self.jump_candidate_x = None
                self.jump_candidate_y = None
                self.jump_candidate_count = 0
            else:
                # 单帧误识别时继续发送上一稳定坐标。
                return (round(self.filtered_car_x, 2),
                        round(self.filtered_car_y, 2))

        # 一阶指数滑动平均：计算量只有加减乘，不使用数组和开方。
        self.filtered_car_x += COORD_EMA_ALPHA * (
            accepted_x - self.filtered_car_x
        )
        self.filtered_car_y += COORD_EMA_ALPHA * (
            accepted_y - self.filtered_car_y
        )

        # 跳变判断始终与上一帧被接受的原始坐标比较，
        # 避免 EMA 自身的滞后被误判成新的大跳变。
        self.last_raw_car_x = accepted_x
        self.last_raw_car_y = accepted_y

        return (round(self.filtered_car_x, 2),
                round(self.filtered_car_y, 2))

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
        self.car_roi = self.get_car_roi_by_tag_corners(map_corners)

        camera.set_normal_mode()

        return map_corners

    @staticmethod
    def get_car_roi_by_tag_corners(map_corners):
        """
        根据四个 AprilTag 角点生成小车识别 ROI。

        OpenMV 的 find_blobs() 使用矩形 ROI，因此这里取四个角点的
        最小外接矩形，并限制在当前 QVGA 图像范围内。
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

    def detect_and_draw_car(self, img, map_corners):
        """
        检测小车坐标。

        当前坐标系由 AprilTag 确定：
            ID0 -> (0, 0)
            ID1 -> (15, 0)
            ID2 -> (15, 11)
            ID3 -> (0, 11)

        成功返回:
            (filtered_x, filtered_y)

        坐标处理：
            原始坐标先经过跳变剔除和轻量 EMA，
            返回值就是最终持续发送给下位机的坐标。

        失败返回:
            None
        """

        if map_corners and self.perspective_matrix is None:
            self.perspective_matrix = self.compute_homography(map_corners)

        if self.perspective_matrix is None:
            return None

        if self.car_roi is None:
            self.car_roi = self.get_car_roi_by_tag_corners(map_corners)

        if self.car_roi is None:
            return None

        head_blobs = img.find_blobs(
            [ELEMENT["car_head"]["threshold"]],
            roi=self.car_roi,
            pixels_threshold=100,
            area_threshold=100,
            merge=False,
            margin=0
        )

        back_blobs = img.find_blobs(
            [ELEMENT["car_back"]["threshold"]],
            roi=self.car_roi,
            pixels_threshold=100,
            area_threshold=100,
            merge=False,
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

        # 轻量坐标滤波。
        # 主程序其他位置不需要修改，所有 O / P / 无条件持续发送
        # 都会自动使用这里返回的稳定坐标。
        return self.filter_car_coord(ax, ay)

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


def capture_weighted_p_coord():
    """
    P 模式三帧加权坐标识别。

    收到一次 P 后重新连续拍摄三张图，权重依次为：
        第一帧：0.2
        第二帧：0.3
        第三帧：0.5

    当前版本不做异常帧剔除；只要任意一帧识别失败，
    本次 P 坐标识别就返回 None，由主循环发送 E5。
    """
    global map_corners

    if map_corners is None:
        return None

    weighted_x = 0.0
    weighted_y = 0.0

    for frame_index in range(len(P_COORD_WEIGHTS)):
        # P 到达后重新拍摄新图，不使用主循环在读取命令前拍下的 img。
        p_img = camera.capture()

        coord = gamesymbol.detect_and_draw_car(p_img, map_corners)

        if coord is None:
            return None

        weight = P_COORD_WEIGHTS[frame_index]
        weighted_x += coord[0] * weight
        weighted_y += coord[1] * weight

    return (round(weighted_x, 2), round(weighted_y, 2))


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

MIN_PIXEL_RATIO = 0.5

DEBUG_PRINT_MAP = False

TAG_FAMILY = image.TAG16H5
REQUIRED_TAG_IDS = (0, 1, 2, 3)

# 内部格子采样半径
# 0.5 是完整格子，0.35 更偏向中心区域，减少边界串色
CELL_SAMPLE_HALF = 0.42


# ==========================
# 小车坐标轻量滤波参数
# ==========================
# True：开启“跳变剔除 + EMA”；False：直接返回原始视觉坐标。
ENABLE_COORD_FILTER = False

# EMA 系数：越大越跟手，越小越平滑。
# 推荐测试范围 0.55 ~ 0.80；当前先使用 0.65。
COORD_EMA_ALPHA = 0.65

# 相邻被接受原始坐标之间允许的最大距离，单位是地图格。
# 这里不计算 sqrt，运行时直接比较距离平方。
COORD_MAX_JUMP = 0.70
COORD_MAX_JUMP_SQ = COORD_MAX_JUMP * COORD_MAX_JUMP

# 当坐标突然跳到远处时，需要连续多少帧确认后才接受。
# 这样单帧误识别不会让小车突然大幅转向。
COORD_JUMP_CONFIRM_FRAMES = 3

# 连续大跳变候选之间允许的距离，单位是地图格。
# 车辆真的快速移动时，相邻候选仍应大致连续。
COORD_JUMP_CONFIRM_RADIUS = 0.70
COORD_JUMP_CONFIRM_RADIUS_SQ = (
    COORD_JUMP_CONFIRM_RADIUS * COORD_JUMP_CONFIRM_RADIUS
)


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
# P 模式三帧加权参数
# ==========================
# 收到 P 后重新拍摄 3 张图，不做异常值剔除。
# 最终坐标 = 第一帧 * 0.2 + 第二帧 * 0.3 + 第三帧 * 0.5。
P_COORD_WEIGHTS = (0.2, 0.3, 0.5)


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
KX_RIGHT = -0.015
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

# # 地图调试

# img = camera.capture()


# if map_corners is None:
#     myuart.writeErr(ERR_NO_MAP_CORNERS)

# visual_ref = gamesymbol.detect_and_draw_car(img, map_corners)

# if visual_ref is None:
#     # print("未识别到小车")
#     myuart.writeErr(ERR_CAR_DETECT_FAIL)

# myuart.writePmode(visual_ref[0], visual_ref[1])

# symbolmap = gamesymbol.detect_grid_color_and_generate_map(img, map_corners)

# if symbolmap is None:
#     # print("地图识别失败")
#     myuart.writeErr(ERR_MAP_DETECT_FAIL)

# theroitic_point = (round(visual_ref[0]), round(visual_ref[1]))

# if 0 <= theroitic_point[0] < GRID_X_NUM and 0 <= theroitic_point[1] < GRID_Y_NUM:
#     symbolmap[theroitic_point[1]][theroitic_point[0]] = ELEMENT["car"]["symbol"]
# else:
#     myuart.writeErr(ERR_CAR_OUT_OF_RANGE)

# print_symbol_map(symbolmap)




# ==========================
# 主程序
# ==========================
while True:
    img = camera.capture()
    cmd = myuart.readCmd()

    if cmd == b'O':
        img = camera.capture()
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

        if DEBUG_PRINT_MAP:
            print_symbol_map(symbolmap)

        myuart.writeMmode(symbolmap)

        # auto_send_coord = True
        # last_coord_send_ms = time.ticks_ms()

        continue

    elif cmd == b'P':
        # img = camera.capture()
        if map_corners is None:
            myuart.writeErr(ERR_NO_MAP_CORNERS)
            continue

        # 收到一次 P 后重新连续拍摄三张图：
        # 第一帧 * 0.2 + 第二帧 * 0.3 + 第三帧 * 0.5。
        coord = capture_weighted_p_coord()

        if coord is None:
            # 三帧中的任意一帧识别失败，本次 P 返回错误码 5。
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

    # coord = gamesymbol.detect_and_draw_car(img, map_corners)
    # if coord is None:
    #     # print("坐标识别失败")
    #     myuart.writeErr(ERR_COORD_DETECT_FAIL)
    #     continue
    # print(f"{coord[0]},{coord[1]}")

    # myuart.writePmode(coord[0], coord[1])
    # continue



    # if auto_send_coord:
    #     now_ms = time.ticks_ms()
    #
    #     if time.ticks_diff(now_ms, last_coord_send_ms) >= COORD_SEND_INTERVAL_MS:
    #         send_current_coord_once()
    #         last_coord_send_ms = now_ms
