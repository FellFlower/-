import sensor
import time
from solver2 import AStarSolver
from machine import UART


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_auto_gain(False)        # 关闭自动增益（颜色检测必备）
sensor.set_auto_whitebal(False)    # 关闭自动白平衡（颜色检测必备）
sensor.skip_frames(time=2000)



uart = UART(2, baudrate=115200)     # 初始化串口 波特率设置为115200

uart.read()

# # 3.18
# ELEMENT_THRESHOLDS = {
#     "wall":   (10, 91, -9, 58, -69, 44),    # 墙壁
#     "box":    (54, 100, -60, 18, 13, 108),     # 箱子
#     "car":    (24, 100, -83, -9, -79, 72),      # 小车
#     "goal":   (37, 61, 65, 103, -75, -37),     # 目标点
#     "bomb":   (24, 52, 30, 81, 6, 41),     # 炸弹
#     "ground": (30, 46, 39, 90, -107, 74),   # 地面
# }

# # 3.19
# ELEMENT_THRESHOLDS = {
#     "wall":   (10, 91, -9, 58, -69, 44),    # 墙壁
#     "box":    (54, 100, -60, 18, 13, 108),     # 箱子
#     "car":    (30, 90, -97, -11, -73, 54),      # 小车
#     "goal":   (37, 61, 65, 103, -75, -37),     # 目标点
#     "bomb":   (24, 52, 30, 81, 6, 41),     # 炸弹
#     "ground": (30, 46, 39, 90, -107, 74),   # 地面
# }

# # 3.26
# ELEMENT_THRESHOLDS = {
#     "wall":   (15, 76, -49, 30, -67, -10),    # 墙壁
#     "box":    (54, 100, -60, 18, 13, 108),     # 箱子
#     "car":    (33, 91, -96, -13, -41, 93),      # 小车
#     "goal":   (38, 68, 63, 106, -68, -29),     # 目标点
#     "bomb":   (24, 52, 30, 81, 6, 41),     # 炸弹
#     "ground": (30, 46, 39, 90, -107, 74),   # 地面
# }

# 3.28
ELEMENT_THRESHOLDS = {
    "wall":   (10, 100, -9, 42, -72, -4),    # 墙壁
    "box":    (54, 100, -60, 18, 13, 108),     # 箱子
    "car_head":    (0, 100, -76, -8, -61, -13),
    "car_back":    (0, 100, -99, -37, -19, 57),
    "goal":   (0, 100, 58, 102, -93, -45),     # 目标点
    "bomb":   (24, 52, 30, 81, 6, 41),     # 炸弹
    "ground": (30, 46, 39, 90, -107, 74),   # 地面
}

Roi = (0,0,320,240) #识别区域
Roi_car = (0,0,320,240)

tl = (23,17)
tr = (298,20)
br = (283,214)
bl = (30,205)
map_corners=[tl,tr,br,bl]

difficulty = 1

ELEMENT_PRIORITY = ["box", "goal", "bomb", "wall", "ground"] #元素识别优先级
STATIC_ELEMENTS = ["wall", "goal", "ground"]    # 首次检测后固定
DYNAMIC_ELEMENTS = ["box", "car"]              # 逐帧实时检测

clock = time.clock()

class GameSymbol:
    def __init__(self):
        self.wall = "#"
        self.box = "$"
        self.car = "@"
        self.goal = "."
        self.bomb = "*"
        self.ground = " "       # 待更改

game_symbol = GameSymbol()


GRID_X_NUM = 16  # x方向格子数
GRID_Y_NUM = 12  # y方向格子数
x_cell_size = 0
y_cell_size = 0


#占比阈值
MIN_PIXEL_RATIO = 0.4  # 元素像素占比≥50%才被视为有效

# ---------------------- 全局变量：保存首次网格信息+静态地图 ----------------------
# 首次初始化后赋值：网格基础参数（所有元素坐标基于此）
grid_base = {
    "cell_size": 0,    # 正方形格子边长
    "grid_min_x": 0,   # 网格左上角x
    "grid_min_y": 0,   # 网格左上角y
}
static_map = []  # 首次检测的静态地图（16行×12列）
real_map = {}

# 记录总共需要侦察的数据个数
box_count = 0
target_count = 0

def detect_and_draw_car(img):
    x=0
    y=0
    car_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["car"]],
        roi=Roi_car,
        pixels_threshold=200,
        # area_threshold=200,
        merge=True,
    )
    if not car_blobs:
        print("未检测到小车！")
        return None
    else:
        x = car_blobs[0].cx()
        y = car_blobs[0].cy()
        # img.draw_circle(x,y,10,(255,0,0))
        coord = calc_virtual_coord((x,y), coord_dict=real_map)
        return coord


def detect_and_draw_wall_border(img):
    """
    识别外墙壁的最小包围矩形，并绘制边框
    :param img: 摄像头帧
    :return: 墙壁矩形 (min_x, min_y, max_x, max_y) 或 None
    """

    # 步骤1：检测所有墙壁色块（LAB模式）
    wall_blobs = img.find_blobs(
        [ELEMENT_THRESHOLDS["wall"]],
        roi=Roi,                # 主要获取区域
        pixels_threshold=200,   # 过滤小噪声色块
        area_threshold=200,     # 过滤小面积噪声
        merge=True,             # 合并所有相邻墙壁色块，便于找外轮廓
        lab=True                # 启用LAB颜色空间（抗光照干扰）
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
    img.draw_rectangle(
        min_x,          # 矩形左上角x
        min_y,          # 矩形左上角y
        wall_width,     # 矩形宽度
        wall_height,    # 矩形高度
        color=(0,255,0),
        thickness=2
    )

    return (min_x, min_y, max_x, max_y)

def detect_grid_color_and_generate_map(img):#检验矩形区域，分割识别

    global static_map,grid_base,box_count,real_map,x_cell_size,y_cell_size

    # if wall_rect == None:
    #     return None

    # min_x, min_y, max_x, max_y = wall_rect
    max_x = sensor.width()   # 初始化为画面最右x
    max_y = sensor.height()  # 初始化为画面最下y
    min_x = 0                # 初始化为画面最左x
    min_y = 0                # 初始化为画面最上y

    wall_w = max_x - min_x
    wall_h = max_y - min_y

    # 1. 计算正方形ROI格子尺寸
    x_cell_size = wall_w / GRID_X_NUM
    y_cell_size = wall_h / GRID_Y_NUM
    # grid_total_w = GRID_X_NUM * x_cell_size #横向分割
    # grid_total_h = GRID_Y_NUM * y_cell_size #竖向分割
    # grid_min_x = min_x + (wall_w - grid_total_w) / 2
    # grid_min_y = min_y + (wall_h - grid_total_h) / 2

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

            # 记录所有格子在摄像头中实际坐标
            roi_cx = int(min_x + x_idx * x_cell_size + x_cell_size/2)
            roi_cy = int(min_y + y_idx * y_cell_size + y_cell_size/2)
            real_map[(x_idx,y_idx)] = (roi_cx,roi_cy)

            # ========== 核心：统计每个元素在ROI内的像素占比 ==========
            element_ratio = {}  # 存储元素:占比
            for element in ELEMENT_PRIORITY:
                blobs = img.find_blobs(
                    [ELEMENT_THRESHOLDS[element]],
                    roi=(roi_x, roi_y, roi_w, roi_h),
                    pixels_threshold=1,
                    area_threshold=1,
                    lab=True
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
                if ratio > max_ratio and ratio >= MIN_PIXEL_RATIO:
                    max_ratio = ratio
                    current_element = element
                    if current_element == "box":
                        box_count += 1



            # ========== 更新地图 + 可视化 ==========
            static_map[y_idx][x_idx] = getattr(game_symbol, current_element)

            # # 绘制无缝网格线
            # if x_idx == 0:
            #     img.draw_line(roi_x, roi_y, roi_x, roi_y + roi_h, color=(0,0,0), thickness=1)
            # if x_idx == GRID_X_NUM - 1:
            #     img.draw_line(roi_x + roi_w, roi_y, roi_x + roi_w, roi_y + roi_h, color=(0,0,0), thickness=1)
            # if y_idx == 0:
            #     img.draw_line(roi_x, roi_y, roi_x + roi_w, roi_y, color=(0,0,0), thickness=1)
            # if y_idx == GRID_Y_NUM - 1:
            #     img.draw_line(roi_x, roi_y + roi_h, roi_x + roi_w, roi_y + roi_h, color=(0,0,0), thickness=1)
            # if x_idx > 0:
            #     img.draw_line(roi_x, roi_y, roi_x, roi_y + roi_h, color=(0,0,0), thickness=1)
            # if y_idx > 0:
            #     img.draw_line(roi_x, roi_y, roi_x + roi_w, roi_y, color=(0,0,0), thickness=1)


    return static_map

# ---------------------- 打印符号化地图 ----------------------
def print_symbol_map(symbol_map):
    """格式化打印符号地图（12行×16列）"""
    if symbol_map == None:
        return None
    print("\n===== 游戏地图（12行×16列） =====")
    for row in symbol_map:
        row_str = "".join(row)#中间加入了空格
        print(row_str)
    print("===================================\n")

def calc_virtual_coord(coord_dict,detected_actual):
    """
    将摄像头像素坐标转换为虚拟地图坐标
    :param detected_actual: (x,y) 像素坐标
    :return: (vx,vy) 虚拟连续坐标
    """

    global grid_base

    if detected_actual ==None or detected_actual == None:
        return None

    x, y = detected_actual

    cell_x = grid_base["cell_size_x"]
    cell_y = grid_base["cell_size_y"]

    #步骤4：找到检测坐标的基准虚拟坐标
    base_virtual = (0,0)
    base_actual = coord_dict[base_virtual]

    # 步骤5：反解连续虚拟坐标（核心，虚拟格=1×1）
    # 真实偏移量 → 虚拟偏移量（支持小数）
    dx_actual = detected_actual[0] - base_actual[0]
    dy_actual = detected_actual[1] - base_actual[1]

    dx_virtual = dx_actual / cell_x  # x方向虚拟偏移（如25像素 → 0.5虚拟格）
    dy_virtual = dy_actual / cell_y  # y方向虚拟偏移（如25像素 → 0.5虚拟格）

    # 最终虚拟坐标（基准虚拟坐标 + 小数偏移）
    virtual_x = base_virtual[0] + dx_virtual
    virtual_y = base_virtual[1] + dy_virtual

    return (round(virtual_x,2), round(virtual_y,2))


def send_uart_list(list):

    total_index = len(list) - 1
    current_index = 0
    initial_tag = True
    for i in range (total_index):
    # 1. 边界判断：如果已经到最后一个坐标，返回None表示结束
        if current_index >= len(list) - 1:
            return None
        # 2. 获取当前坐标和下一个坐标
        # 等待串口发送消息
        # 第一次发送不等待
        while(True):
            uart_data = uart.any()
            if uart_data:
                uart_str = uart.read(uart_data).decode()
                if uart_str == "O":
                    initial_tag = False
                    uart.read()
                    break
                else:
                    continue
            elif initial_tag:
                initial_tag = False
                break
            else:
                continue

        current_pos = list[current_index]
        next_pos = list[current_index + 1]
        current_x, current_y = current_pos
        next_x, next_y = next_pos

        # 3. 判断移动方向（仅支持纯X或纯Y方向，符合你的需求）
        if current_y == next_y:  # Y坐标不变，X方向移动
            Mode = 1
        elif current_x == next_x:  # X坐标不变，Y方向移动
            Mode = 2

        x = next_x
        y = next_y
        # else:  # 若出现斜向移动，返回错误标识（可根据需求调整）
        #     action = "err"
        current_index += 1
        # img = sensor.snapshot().rotation_corr(corners = map_corners).lens_corr(0.3)
        # visual_ref = detect_and_draw_car(img)
        # posx,posy  = visual_ref
        posx,posy = 0,0
        uart.write("@"+f"{Mode},{x},{y},{posx},{posy}"+"#")



# 等待初始化
time.sleep(2)

###### 1.传入地图

clock.tick()
img = sensor.snapshot().lens_corr(0.6).rotation_corr(corners = map_corners)
symbolmap = detect_grid_color_and_generate_map(img)
print_symbol_map(symbolmap)
visual_ref = detect_and_draw_car(img)
print(f"实际坐标{visual_ref}")

theroitic_point = (round(visual_ref[0]),round(visual_ref[1]))
print(f"理论坐标{theroitic_point}")

AStar = AStarSolver(raw_map=symbolmap,start_point=theroitic_point,difficulty=difficulty)

# 到达指定玩家初始位置
start_point = theroitic_point
x = start_point[0]
y = start_point[1]

uart.write("@"+f"2,1,{y},0,0"+"#")

# while(True):
#     uart_str = uart.any()
#     if uart_str == "o":
#         break

# uart.write("@"+f"2,{start_point[0]},{start_point[1]},{start_point[0]},{visual_ref[1]}"+"#")
# ###### 2.到目标点和箱子处查看图案，标定序列

# 定义基本量
order = 2*box_count
progress = 0
match_target = {}
match_box = {}

target_order = [0,2,1]
box_order = [0,1,2]



# if difficulty != 1:
#     while (order > 0): # 最后一次侦察结束，order = 0,循环结束
#         clock.tick()
#         img = sensor.snapshot().rotation_corr(corners = map_corners).lens_corr(0.4)
#         uart_str = uart.any()
#         # detect_and_draw_car(img)
#         # 待车传回信息之后，进行下一条路径指示
#         if uart.any():
#             uart.read()
#             if order > box_count:
#                 ifbox=True
#                 iftarget=False
#                 temp_order=2*box_count-order
#             elif order <= box_count:
#                 ifbox=False
#                 iftarget=True
#                 temp_order = box_count-order

#             Pre_path,invest_point = AStar.investigate(ifbox=ifbox,iftarget=iftarget,order=temp_order)
#             Pre_Path = AStar.extract_turning_points(Pre_path)
#             send_uart_list(Pre_Path)
#             # 等待小车解读图像
#             while True:
#                 # 读取串口数据(确认目标)
#                 if uart.any():
#                     uart_str = uart.read().strip()
#                     if order > box_count:
#                         match_box[2*box_count-order] = int(uart_str)
#                         uart.write("detect mission complete")
#                         order -= 1
#                         uart.read()
#                         break

#                     elif order <= box_count:
#                         match_target[box_count-order] = int(uart_str)
#                         uart.write("detect mission complete")
#                         order -= 1
#                         uart.read()
#                         break

#     value_to_key2 = {val: key for key, val in match_box.items()}
#     match = {key1: value_to_key2[val] for key1, val in match_target.items()}
#     for i in range(box_count):
#         box_order.append(match[i])
#     print(f"序列已确认！box顺序：{box_order}")

##### 3.初始寻路逻辑 ######
finall_path = AStar.solve(box_order=box_order,target_order=target_order)
keys = list(finall_path.keys())
total_steps = len(keys)
keys_sorted = sorted(keys)
process = 0

while (process < total_steps):
    # clock.tick()
    # img = sensor.snapshot()
    # 接收到准备行动指令
    uart_data = uart.any()
    if uart_data:
        uart_str = uart.read(uart_data).decode()
        if uart_str == "O":
            send_path = AStar.extract_turning_points(finall_path[process])
            send_uart_list(send_path)
            process += 1
            uart.read()
        else:
            continue


uart.read()
print("任务结束")







