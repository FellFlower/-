from pyb import LED
from machine import UART
import sensor, time, seekfree, math


uart = UART(12, baudrate=115200)

red = LED(1)
green = LED(2)
blue = LED(3)
white = LED(4)

# ==========================
# 等待主控发送 O
# ==========================
# while True:
#     uart_num = uart.any()
#     if uart_num:
#         uart_str = uart.read(uart_num)

#         # OpenMV 里 uart.read() 一般返回 bytes
#         if uart_str and (b"O" in uart_str):
#             break
#         else:
#             continue


# ==========================
# 颜色阈值
# ==========================
ELEMENT_THRESHOLDS = {
    "wall":   (30, 89, -7, 47, -79, 7),       # 墙壁
    "box":    (17, 95, -42, 34, 21, 93),      # 箱子
    "car_head": (65, 93, -51, -6, -49, 2),  # 车头
    "car_back": (63, 91, -84, -56, -1, 86),   # 车尾
    "goal":   (17, 88, 64, 107, -86, -29),    # 目标点
    "bomb":   (24, 77, 24, 95, -27, 86),      # 炸弹
}

map_corners = [
        (10 , 9),
        (316 , 5),
        (319 , 229),
        (12 , 230),
]

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
# 全局变量
# ==========================
grid_base = {
    "cell_size_x": 0,
    "cell_size_y": 0,
    "grid_min_x": 0,
    "grid_min_y": 0,
}

static_map = []

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

def apply_homography(matrix, x, y):
    """应用透视矩阵，返回归一化坐标 (0~1)"""
    A, B, C, D, E, F, G, H_m, I_m = matrix
    w = G * x + H_m * y + I_m
    if w == 0:
        w = 0.0001
    x_norm = (A * x + B * y + C) / w
    y_norm = (D * x + E * y + F) / w
    return x_norm, y_norm

def radial_distortion_correct(x, y):
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


# 全局存储透视矩阵，避免每帧重复计算
PERSPECTIVE_MATRIX = None

def detect_and_draw_car(img, map_corners):
    global car_angle
    global PERSPECTIVE_MATRIX

    # 如果传入了新角点，更新透视矩阵
    if map_corners:
        PERSPECTIVE_MATRIX = compute_homography(map_corners)

    if PERSPECTIVE_MATRIX is None:
        return None, None

    head_blobs = img.find_blobs([ELEMENT_THRESHOLDS["car_head"]], roi=Roi_car, pixels_threshold=100, area_threshold=100, merge=True, margin=0)
    back_blobs = img.find_blobs([ELEMENT_THRESHOLDS["car_back"]], roi=Roi_car, pixels_threshold=100, area_threshold=100, merge=True, margin=0)

    if not head_blobs or not back_blobs:
        # print("未检测到小车！")
        car_angle = None
        return None, None
    else:
        b1 = max(head_blobs, key=lambda b: b.pixels())   # 车头
        b2 = max(back_blobs, key=lambda b: b.pixels())   # 车尾

        # 获取在原始畸变图像中的像素坐标
        x1, y1 = b1.cx(), b1.cy()
        x2, y2 = b2.cx(), b2.cy()

        # 核心：将车头和车尾分别映射到 0~1 的归一化地图空间
        nx1, ny1 = apply_homography(PERSPECTIVE_MATRIX, x1, y1)
        nx2, ny2 = apply_homography(PERSPECTIVE_MATRIX, x2, y2)

        # 乘以地图网格数，直接得到真实世界的虚拟坐标 (完美矫正)
        ax1 = nx1 * GRID_X_NUM
        ay1 = ny1 * GRID_Y_NUM
        ax2 = nx2 * GRID_X_NUM
        ay2 = ny2 * GRID_Y_NUM

        # 1. 小车坐标为真实世界车头车尾的中点
        ax = (ax1 + ax2) / 2 - 0.5
        ay = (ay1 + ay2) / 2 - 0.5

        ax, ay = radial_distortion_correct(ax, ay)

        # 2. 计算完美矫正后的车身角度 (用真实空间下的相对位置算角度)
        dx = ax2 - ax1
        dy = ay2 - ay1

        if dx == 0 and dy == 0:
            car_angle = None
        else:
            car_angle = math.degrees(math.atan2(-dx, dy))

            if car_angle > 180:
                car_angle -= 360
            elif car_angle < -180:
                car_angle += 360

            car_angle = round(car_angle, 2)

        return (round(ax, 2), round(ay, 2)), car_angle


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


def detect_and_draw_wall_border(img,roi):
    """
    识别外墙壁的最小包围矩形，并绘制边框
    :param img: 摄像头帧
    :return: 墙壁矩形 (min_x, min_y, max_x, max_y) 或 None
    """

    # 步骤1：检测所有墙壁色块（LAB模式）
    wall_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["wall"]],
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
    min_x = sensor.width()   # 初始化为画面最右x
    min_y = sensor.height()  # 初始化为画面最下y
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
    max_x = min(sensor.width() - 1, max_x)
    max_y = min(sensor.height() - 1, max_y)

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

    return (min_x, min_y, max_x, max_y)

def detect_grid_color_and_generate_map(img):#检验矩形区域，分割识别

    global static_map,grid_base

    max_x = sensor.width()   # 初始化为画面最右x
    max_y = sensor.height()  # 初始化为画面最下y
    min_x = 0                # 初始化为画面最左x
    min_y = 0                # 初始化为画面最上y
    wall_w = max_x - min_x
    wall_h = max_y - min_y

    # 1. 计算正方形ROI格子尺寸
    x_cell_size = wall_w / GRID_X_NUM
    y_cell_size = wall_h / GRID_Y_NUM

    grid_base["grid_min_x"] = min_x
    grid_base["grid_min_y"] = min_y
    # grid_base["cell_size_x"] = x_cell_size
    # grid_base["cell_size_y"] = y_cell_size

    # 2. 初始化符号地图(全为地面)
    static_map = [[game_symbol.ground for _ in range(GRID_X_NUM)] for _ in range(GRID_Y_NUM)]

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
            roi_w = min(roi_w, sensor.width() - roi_x)
            roi_h = min(roi_h, sensor.height() - roi_y)
            roi_total_pixels = roi_w * roi_h  # ROI总像素数
            if roi_total_pixels == 0:
                continue

            # ========== 核心：统计每个元素在ROI内的像素占比 ==========
            element_ratio = {}  # 存储元素:占比
            for element in ELEMENT_PRIORITY:
                blobs = img.find_blobs(
                    [ELEMENT_THRESHOLDS[element]],
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

    # ==========================
    # 径向畸变矫正
    # ==========================
    virtual_x, virtual_y = radial_distortion_correct(virtual_x, virtual_y)

    return (round(virtual_x, 2), round(virtual_y, 2))


def position_adjustment():
    pass

# ==========================
# 摄像头初始化
# ==========================
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
sensor.skip_frames(time=1000)


# ==========================
# 初始化地图并发送
# ==========================
clock.tick()
while(True):
    img = sensor.snapshot().lens_corr(0.35)
    uart_num = uart.any()
    if uart_num:
        uart_str = uart.read(uart_num)
        if uart_str and (b"O" in uart_str):
            x1,y1,x3,y3 = detect_and_draw_wall_border(img,Roi)
            co1 = (x1+1,y1+1)
            co2 = (x3-1,y1+1)
            co3 = (x3-1,y3-1)
            co4 = (x1+1,y3-1)
            map_corners=[co1,co2,co3,co4]

            visual_ref,angle = detect_and_draw_car(img,map_corners)

            uart.write("&P" + f"{visual_ref[0]}" + "," + f"{visual_ref[1]}" + "%")
            # map_corners = get_perspective_corners(img,Roi)
            img = sensor.snapshot().lens_corr(0.35).rotation_corr(corners = map_corners)

            symbolmap = detect_grid_color_and_generate_map(img)

            if visual_ref is not None:
                theroitic_point = (round(visual_ref[0]), round(visual_ref[1]))
                print(f"理论坐标{theroitic_point}")

                if 0 <= theroitic_point[0] < GRID_X_NUM and 0 <= theroitic_point[1] < GRID_Y_NUM:
                    symbolmap[theroitic_point[1]][theroitic_point[0]] = "@"
                else:
                    print("小车理论坐标越界，不写入地图")
            else:
                print("未识别到小车，不写入 @")

            map_str_data = ""

            for row in symbolmap:
                for element in row:
                    map_str_data += element

            frame = "&M" + map_str_data + "%"
            uart.write(frame)
            break

# ==========================
# 主循环
# 收到 P 后返回小车坐标
# ==========================
while True:
    img = sensor.snapshot().lens_corr(0.35)
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


            # lcd.show_image(img, 320, 240, zoom=0)
            coord,angle = detect_and_draw_car(img,map_corners)
            if coord is None:
                print("坐标识别失败")
                uart.write("&PERR%")
            else:
                uart.write("&P" + f"{coord[0]}" + "," + f"{coord[1]}" + "%")

            continue

        else:
            continue

