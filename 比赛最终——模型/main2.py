import sensor, time, tf, gc
from machine import UART

uart = UART(12, baudrate=115200)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_framerate(60)
sensor.set_brightness(200)
sensor.set_windowing(320, 240)
sensor.skip_frames(time=2000)

# ==================================================
# 模型
# ==================================================
detect_net = tf.load("yolo3_v2.tflite", load_to_fb=True)
cartoon_net = tf.load("model7.0.tflite", load_to_fb=True)
num_net = tf.load("newnum.tflite", load_to_fb=True)

cartoon_labels = [x.rstrip() for x in open("/sd/characters.txt")]
num_labels = [x.rstrip() for x in open("/sd/numbers.txt")]

# ==================================================
# 参数
# ==================================================
FRAME_COUNT = 2
YOLO1_THRESHOLD = 0.50
YOLO2_THRESHOLD = 0.50
CLASSIFY_THRESHOLD = 0.70
MIN_VOTES = 2
MAX_PENDING_CMDS = 4

# ==================================================
#             L2      C2      R2
#                     C1
#                     车
# ==================================================
ROI_C1 = (36, 1, 224, 218)
ROI_L2 = (0, 22, 136, 148)
ROI_C2 = (74, 22, 137, 149)
ROI_R2 = (166, 21, 135, 150)

# ==================================================
# 串口协议
#
# 左：
# 0 = 不识别
# 1 = L2人物
# 2 = L2数字
#
# 中：
# 0 = 不识别
# 1 = C1人物
# 2 = C1数字
# 3 = C2人物
# 4 = C2数字
#
# 右：
# 0 = 不识别
# 1 = R2人物
# 2 = R2数字
#
# cmd = left * 35 + center * 5 + right + 1
#
# 有效cmd范围：1~93
# ==================================================


def decode_side(v, left):
    if v == 0:
        return None
    if v not in (1, 2):
        return None

    roi = ROI_L2 if left else ROI_R2
    return roi, v


def decode_center(v):
    if v == 0:
        return None
    if v == 1:
        return ROI_C1, 1
    if v == 2:
        return ROI_C1, 2
    if v == 3:
        return ROI_C2, 1
    if v == 4:
        return ROI_C2, 2

    return None


def decode_cmd(cmd):
    if cmd < 1 or cmd > 93:
        return None

    v = cmd - 1

    left = v // 35
    center = (v % 35) // 5
    right = v % 5

    if left > 2 or center > 4 or right > 2:
        return None

    return [
        decode_side(left, True),
        decode_center(center),
        decode_side(right, False)
    ]


def get_label_id(labels, index):
    try:
        value = int(labels[index])

        if 0 <= value <= 9:
            return value

    except Exception:
        pass

    return index if index <= 9 else 0


# ==================================================
# YOLO
# 对传入图像进行一次YOLO
# 只保留置信度最高的框
# ==================================================
def yolo_detect(crop, threshold):
    best_rect = None
    best_score = 0.0

    detections = tf.detect(detect_net, crop)

    img_w = crop.width()
    img_h = crop.height()

    for obj in detections:
        try:
            x1, y1, x2, y2, label, score = obj
        except Exception:
            continue

        if score < threshold:
            continue

        x1 = int(x1 * img_w)
        y1 = int(y1 * img_h)
        x2 = int(x2 * img_w)
        y2 = int(y2 * img_h)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > img_w:
            x2 = img_w
        if y2 > img_h:
            y2 = img_h

        w = x2 - x1
        h = y2 - y1

        if w <= 1 or h <= 1:
            continue

        if score > best_score:
            best_score = score
            best_rect = (x1, y1, w, h)

    return best_rect


# ==================================================
# 分类
#
# mode = 1：人物
# mode = 2：数字
# ==================================================
def classify_target(target, mode):
    net = cartoon_net if mode == 1 else num_net
    labels = cartoon_labels if mode == 1 else num_labels

    best_id = None
    best_name = None
    best_prob = 0.0

    for obj in tf.classify(
        net,
        target,
        min_scale=1,
        scale_mul=0.8,
        x_overlap=0.5,
        y_overlap=0.5
    ):
        pred = list(obj.output())

        if not pred:
            continue

        index = pred.index(max(pred))
        prob = pred[index]

        if prob > best_prob:
            best_prob = prob
            best_id = get_label_id(labels, index)
            best_name = labels[index]

    return best_id, best_name, best_prob


# ==================================================
# 单个ROI完整识别
# ==================================================
def recognize_roi(img, roi, mode):
    gc.collect()

    # ===== 手动ROI =====
    manual_crop = img.copy(
        roi=roi,
        copy_to_fb=False
    )

    # ===== YOLO1 =====
    yolo1_rect = yolo_detect(
        manual_crop,
        YOLO1_THRESHOLD
    )

    if yolo1_rect is None:
        del manual_crop
        gc.collect()
        return None, None, 0.0

    # 裁YOLO1
    yolo1_crop = manual_crop.copy(
        roi=yolo1_rect,
        copy_to_fb=False
    )

    # 后面已经不需要manual_crop，立即释放
    del manual_crop
    gc.collect()

    # ===== YOLO2 =====
    yolo2_rect = yolo_detect(
        yolo1_crop,
        YOLO2_THRESHOLD
    )

    if yolo2_rect is None:
        del yolo1_crop
        gc.collect()
        return None, None, 0.0

    # 裁YOLO2
    final_target = yolo1_crop.copy(
        roi=yolo2_rect,
        copy_to_fb=False
    )

    # 后面已经不需要yolo1_crop
    del yolo1_crop
    gc.collect()

    # ===== 分类 =====
    label_id, label_name, class_prob = classify_target(
        final_target,
        mode
    )

    del final_target
    gc.collect()

    if class_prob < CLASSIFY_THRESHOLD:
        return None, None, class_prob

    return label_id, label_name, class_prob


# ==================================================
# 多帧投票
# ==================================================
def vote_result(results):
    stats = {}

    for label_id, label_name, prob in results:
        if label_id is None:
            continue

        if label_id not in stats:
            stats[label_id] = [0, 0.0, label_name]

        stats[label_id][0] += 1

        if prob > stats[label_id][1]:
            stats[label_id][1] = prob
            stats[label_id][2] = label_name

    if not stats:
        return None, 0

    best_id = None
    best_count = -1
    best_prob = -1.0

    for label_id, value in stats.items():
        count = value[0]
        prob = value[1]

        if count > best_count or (
            count == best_count and prob > best_prob
        ):
            best_id = label_id
            best_count = count
            best_prob = prob

    return best_id, best_count


# ==================================================
# 执行一条完整串口识别指令
# ==================================================
def process_command(cmd):
    tasks = decode_cmd(cmd)

    if tasks is None:
        return

    results = [[], [], []]

    # ----------------------------------------------
    # 采集FRAME_COUNT帧
    # ----------------------------------------------
    for frame in range(FRAME_COUNT):
        img = sensor.snapshot().rotation_corr(
            z_rotation=180,
            zoom=1.0
        )

        # ------------------------------------------
        # L / C / R
        # ------------------------------------------
        for slot in range(3):
            task = tasks[slot]

            if task is None:
                continue

            roi, mode = task

            result = recognize_roi(
                img,
                roi,
                mode
            )

            results[slot].append(result)

            # # 识别过程中顺便读取UART
            # # 防止后续指令长时间滞留在接收缓存
            # read_uart_commands()

        del img
        gc.collect()

    # ----------------------------------------------
    # 投票
    # ----------------------------------------------
    output = [None, None, None]

    for slot in range(3):
        if tasks[slot] is None:
            continue

        label_id, votes = vote_result(
            results[slot]
        )

        if votes < MIN_VOTES:
            label_id = None

        output[slot] = label_id

    # ----------------------------------------------
    # 与之前上位机协议保持一致
    #
    # 只发送参与识别的位置
    #
    # 例如：
    # L=3 C=5 R=7
    # -> &357%
    #
    # 如果只有C：
    # -> &5%
    #
    # 无有效分类结果：
    # -> 0
    # ----------------------------------------------
    send_str = ""

    for slot in range(3):
        if tasks[slot] is None:
            continue

        if output[slot] is None:
            output[slot] = 0

        send_str += str(output[slot])

    uart.write("&" + send_str + "%")

    del results
    del output
    del tasks

    gc.collect()


# ==================================================
# UART指令队列
# ==================================================
pending_cmds = []


def read_uart_commands():
    n = uart.any()

    if n <= 0:
        return

    data = uart.read(n)

    if not data:
        return

    for cmd in data:
        if decode_cmd(cmd) is None:
            continue

        if len(pending_cmds) < MAX_PENDING_CMDS:
            pending_cmds.append(cmd)


# ==================================================
# 主循环
# ==================================================
while True:
    # 先检查串口
    read_uart_commands()

    # ----------------------------------------------
    # 收到有效指令才开始YOLO+分类
    # ----------------------------------------------
    if pending_cmds:
        cmd = pending_cmds.pop(0)

        gc.collect()

        process_command(cmd)

        gc.collect()

    # ----------------------------------------------
    # 常态下保持相机持续采集
    # 不执行YOLO
    # 不执行分类
    # 不draw
    # 不print
    # ----------------------------------------------
    else:
        pass
        img = sensor.snapshot().rotation_corr(
            z_rotation=180,
            zoom=1.0
        )
