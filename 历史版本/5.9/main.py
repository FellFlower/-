from pyb import LED
from machine import UART
import sensor, time, seekfree


uart = UART(2, baudrate=115200)

red = LED(1)
green = LED(2)
blue = LED(3)
white = LED(4)


# ==========================
# 等待主控发送 O
# ==========================
while True:
    uart_num = uart.any()
    if uart_num:
        uart_str = uart.read(uart_num)

        # OpenMV 里 uart.read() 一般返回 bytes
        if uart_str and (b"O" in uart_str):
            break
        else:
            continue


# ==========================
# 颜色阈值
# ==========================
ELEMENT_THRESHOLDS = {
    "wall":   (30, 89, -7, 47, -79, 7),       # 墙壁
    "box":    (17, 95, -42, 34, 21, 93),      # 箱子
    "car_head": (0, 100, -55, -11, -59, -8),  # 车头
    "car_back": (0, 100, -89, -47, -8, 88),   # 车尾
    "goal":   (17, 88, 64, 107, -86, -29),    # 目标点
    "bomb":   (24, 77, 24, 95, -27, 86),      # 炸弹
}


Roi = (0, 0, 320, 240)
Roi_car = (0, 0, 320, 240)

ELEMENT_PRIORITY = ["box", "goal", "bomb", "wall"]

clock = time.clock()


class GameSymbol:
    def __init__(self):
        self.wall = "#"
        self.box = "$"
        self.car = "@"
        self.goal = "."
        self.bomb = "*"
        self.ground = "-"


game_symbol = GameSymbol()


GRID_X_NUM = 16
GRID_Y_NUM = 12


# 占比阈值
MIN_PIXEL_RATIO = 0.4


# ==========================
# 粗扫参数
# ==========================
ROUGH_WALL_RATIO = 0.05
ROUGH_ROI_PAD = 18


# ==========================
# 全局变量
# ==========================
grid_base = {
    "cell_size_x": 0,
    "cell_size_y": 0,
    "grid_min_x": 0,
    "grid_min_y": 0,
}

static_map = []


def detect_and_draw_car(img):
    x = 0
    y = 0

    head_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["car_head"]],
        roi=Roi_car,
        pixels_threshold=80,
        merge=True,
        margin=10
    )

    back_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["car_back"]],
        roi=Roi_car,
        pixels_threshold=80,
        merge=True,
        margin=10
    )

    if not head_blobs or not back_blobs:
        print("未检测到小车！")
        return None
    else:
        b1 = head_blobs[0]
        x1 = b1.x() + b1.w() / 2
        y1 = b1.y() + b1.h() / 2

        b2 = back_blobs[0]
        x2 = b2.x() + b2.w() / 2
        y2 = b2.y() + b2.h() / 2

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        img.draw_circle(int(x1), int(y1), 4, color=(0, 255, 0), thickness=2)
        img.draw_circle(int(x2), int(y2), 4, color=(0, 0, 255), thickness=2)
        img.draw_circle(int(x), int(y), 4, color=(255, 255, 0), thickness=2)

        coord = calc_virtual_coord_2((x, y))

        if coord is None:
            return None

        ax, ay = coord
        return (ax, ay)


def get_perspective_corners(img, roi):
    """
    使用最小外接旋转矩形获取完美的四个角点，并按顺时针(TL, TR, BR, BL)排序
    """
    wall_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["wall"]],
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


def detect_and_draw_wall_border(img, roi):
    """
    原来的兜底方法：识别外墙壁的最小包围矩形。
    """
    wall_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["wall"]],
        roi=roi,
        pixels_threshold=300,
        area_threshold=300,
        merge=True,
        lab=True,
    )

    if not wall_blobs:
        print("未检测到墙壁色块！")
        return None

    min_x = sensor.width()
    min_y = sensor.height()
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
    max_x = min(sensor.width() - 1, max_x)
    max_y = min(sensor.height() - 1, max_y)

    wall_width = max_x - min_x
    wall_height = max_y - min_y

    if wall_width < 20 or wall_height < 20:
        print("检测到的墙壁区域过小，无效！")
        return None

    img.draw_rectangle(
        min_x,
        min_y,
        wall_width,
        wall_height,
        color=(255, 0, 0),
        thickness=2
    )

    return (min_x, min_y, max_x, max_y)


def detect_grid_color_and_generate_map(img):
    """
    原来的地图分割识别逻辑，不改变。
    """
    global static_map, grid_base

    max_x = sensor.width()
    max_y = sensor.height()
    min_x = 0
    min_y = 0

    wall_w = max_x - min_x
    wall_h = max_y - min_y

    x_cell_size = wall_w / GRID_X_NUM
    y_cell_size = wall_h / GRID_Y_NUM

    grid_base["grid_min_x"] = min_x
    grid_base["grid_min_y"] = min_y
    grid_base["cell_size_x"] = x_cell_size
    grid_base["cell_size_y"] = y_cell_size

    static_map = [[game_symbol.ground for _ in range(GRID_X_NUM)] for _ in range(GRID_Y_NUM)]

    for y_idx in range(GRID_Y_NUM):
        for x_idx in range(GRID_X_NUM):

            roi_x = int(min_x + x_idx * x_cell_size)
            roi_y = int(min_y + y_idx * y_cell_size)
            roi_w = int(x_cell_size)
            roi_h = int(y_cell_size)

            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, sensor.width() - roi_x)
            roi_h = min(roi_h, sensor.height() - roi_y)

            roi_total_pixels = roi_w * roi_h
            if roi_total_pixels == 0:
                continue

            element_ratio = {}

            for element in ELEMENT_PRIORITY:
                blobs = img.find_blobs(
                    [ELEMENT_THRESHOLDS[element]],
                    roi=(roi_x, roi_y, roi_w, roi_h),
                    pixels_threshold=90,
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

            static_map[y_idx][x_idx] = getattr(game_symbol, current_element)

    return static_map


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


def calc_virtual_coord_2(detected_actual):
    """
    将摄像头像素坐标转换为虚拟地图坐标。
    """
    global grid_base

    if detected_actual is None:
        return None

    x, y = detected_actual

    cell_x = grid_base["cell_size_x"]
    cell_y = grid_base["cell_size_y"]

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

    return (round(virtual_x, 2), round(virtual_y, 2))


def position_adjustment():
    pass


# ==========================
# 新增：粗扫 rough_roi
# ==========================
def corners_from_roi(roi):
    x = roi[0]
    y = roi[1]
    w = roi[2]
    h = roi[3]

    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]


def detect_rough_roi_by_grid(img):
    """
    用 detect_grid_color_and_generate_map 的思路先粗扫整张图，
    找出墙壁大概范围 rough_roi。
    """
    min_x = 0
    min_y = 0
    max_x = sensor.width()
    max_y = sensor.height()

    wall_w = max_x - min_x
    wall_h = max_y - min_y

    x_cell_size = wall_w / GRID_X_NUM
    y_cell_size = wall_h / GRID_Y_NUM

    rough_min_x = sensor.width()
    rough_min_y = sensor.height()
    rough_max_x = 0
    rough_max_y = 0
    rough_found = False

    for y_idx in range(GRID_Y_NUM):
        for x_idx in range(GRID_X_NUM):

            roi_x = int(min_x + x_idx * x_cell_size)
            roi_y = int(min_y + y_idx * y_cell_size)
            roi_w = int(x_cell_size)
            roi_h = int(y_cell_size)

            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, sensor.width() - roi_x)
            roi_h = min(roi_h, sensor.height() - roi_y)

            roi_total_pixels = roi_w * roi_h
            if roi_total_pixels <= 0:
                continue

            wall_blobs = img.find_blobs(
                [ELEMENT_THRESHOLDS["wall"]],
                roi=(roi_x, roi_y, roi_w, roi_h),
                pixels_threshold=40,
                area_threshold=20,
                merge=True,
                margin=2,
                lab=True
            )

            wall_pixels = 0
            for b in wall_blobs:
                wall_pixels += b.pixels()

            wall_ratio = wall_pixels / roi_total_pixels

            if wall_ratio >= ROUGH_WALL_RATIO:
                rough_found = True

                rough_min_x = min(rough_min_x, roi_x)
                rough_min_y = min(rough_min_y, roi_y)
                rough_max_x = max(rough_max_x, roi_x + roi_w)
                rough_max_y = max(rough_max_y, roi_y + roi_h)

    if not rough_found:
        print("粗扫没有找到地图区域")
        return None

    rough_min_x = max(0, rough_min_x - ROUGH_ROI_PAD)
    rough_min_y = max(0, rough_min_y - ROUGH_ROI_PAD)
    rough_max_x = min(sensor.width() - 1, rough_max_x + ROUGH_ROI_PAD)
    rough_max_y = min(sensor.height() - 1, rough_max_y + ROUGH_ROI_PAD)

    rough_roi = (
        int(rough_min_x),
        int(rough_min_y),
        int(rough_max_x - rough_min_x),
        int(rough_max_y - rough_min_y)
    )

    img.draw_rectangle(
        rough_roi[0],
        rough_roi[1],
        rough_roi[2],
        rough_roi[3],
        color=(0, 255, 0),
        thickness=2
    )

    return rough_roi


def get_map_corners_two_stage(img0):
    """
    两阶段获取地图角点：
    1. 先用网格粗扫找到 rough_roi
    2. 再在 rough_roi 里面用 get_perspective_corners_v2 精找角点
    3. 如果 v2 失败，用 rough_roi 矩形兜底
    4. 如果 rough_roi 也失败，用原外接框方法兜底
    """
    rough_roi = detect_rough_roi_by_grid(img0)

    if rough_roi is not None:
        print("rough_roi =", rough_roi)

        corners = get_perspective_corners_v2(img0, rough_roi)

        if corners is not None:
            print("v2 角点成功")
            print("corners =", corners)
            return corners

        print("v2 角点失败，使用 rough_roi 矩形兜底")
        corners = corners_from_roi(rough_roi)
        print("corners =", corners)
        return corners

    print("rough_roi 失败，使用原外接框方法兜底")

    border = detect_and_draw_wall_border(img0, Roi)

    if border is not None:
        x1, y1, x3, y3 = border

        corners = [
            (x1, y1),
            (x3, y1),
            (x3, y3),
            (x1, y3)
        ]

        print("外接框兜底成功")
        print("corners =", corners)
        return corners

    print("角点获取彻底失败")
    return None


# ==========================
# 摄像头初始化
# ==========================
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
sensor.skip_frames(time=200)


# ==========================
# 初始化地图并发送
# ==========================
clock.tick()
img = sensor.snapshot().lens_corr(0.35)
x1,y1,x3,y3 = detect_and_draw_wall_border(img,Roi)
co1 = (x1+1,y1+1)
co2 = (x3-1,y1+1)
co3 = (x3-1,y3-1)
co4 = (x1+1,y3-1)
map_corners=[co1,co2,co3,co4]
img = sensor.snapshot().lens_corr(0.35).rotation_corr(corners = map_corners)

symbolmap = detect_grid_color_and_generate_map(img)

visual_ref = detect_and_draw_car(img)

print(f"实际坐标{visual_ref}")

if visual_ref is not None:
    theroitic_point = (round(visual_ref[0]), round(visual_ref[1]))
    print(f"理论坐标{theroitic_point}")

    if 0 <= theroitic_point[0] < GRID_X_NUM and 0 <= theroitic_point[1] < GRID_Y_NUM:
        symbolmap[theroitic_point[1]][theroitic_point[0]] = "@"
    else:
        print("小车理论坐标越界，不写入地图")
else:
    print("未识别到小车，不写入 @")

print_symbol_map(symbolmap)


map_str_data = ""

for row in symbolmap:
    for element in row:
        map_str_data += element

frame = "&M" + map_str_data + "%"

uart.write(frame)


lcd = seekfree.IPS200(3)
lcd.full()
lcd.show_image(img, 320, 240, zoom=0)


# ==========================
# 主循环
# 收到 P 后返回小车坐标
# ==========================
while True:
    uart_num = uart.any()

    if uart_num:
        uart_str = uart.read(uart_num)

        if uart_str and (b"P" in uart_str):
            img = sensor.snapshot().lens_corr(0.35)
            x1,y1,x3,y3 = detect_and_draw_wall_border(img,Roi)
            co1 = (x1+1,y1+1)
            co2 = (x3-1,y1+1)
            co3 = (x3-1,y3-1)
            co4 = (x1+1,y3-1)
            map_corners=[co1,co2,co3,co4]

            # map_corners = get_perspective_corners(img,Roi)

            img = sensor.snapshot().lens_corr(0.35).rotation_corr(corners = map_corners)

            coord = detect_and_draw_car(img)
            lcd.show_image(img, 320, 240, zoom=0)

            if coord is None:
                print("坐标识别失败")
                uart.write("&PERR%")
            else:
                uart.write("&P" + f"{coord[0]}" + "," + f"{coord[1]}" + "%")

            continue

        else:
            continue
