import sensor, time, tf, gc
from machine import UART

# ==================================================
# UART
# ==================================================
# 优先尝试增加RX/TX缓冲区
try:
    uart = UART(
        12,
        baudrate=115200,
        rxbuf=64,
        txbuf=64,
        timeout=0
    )
except Exception:
    uart = UART(12, baudrate=115200)

# UART中断标志
uart_irq_flag = 0
UART_IRQ_ENABLED = False

# ==================================================
# 调试开关
# ==================================================
DEBUG_UART = True
DEBUG_DETAIL = False

# 是否向RT1064真正返回识别结果
UART_TX_ENABLE = True

# ==================================================
# UART中断
# 中断里面绝对不要：
# print
# uart.read
# append
# gc.collect
# 只置标志
# ==================================================
def uart_irq_handler(line):
    global uart_irq_flag
    uart_irq_flag = 1

# 优先RXIDLE
try:
    uart.irq(
        handler=uart_irq_handler,
        trigger=UART.IRQ_RXIDLE,
        hard=True
    )
    UART_IRQ_ENABLED = True
except Exception:
    # 兼容部分逐飞/OpenART固件
    try:
        uart.irq(
            handler=uart_irq_handler,
            trigger=UART.IRQ_RX_ANY,
            hard=True
        )
        UART_IRQ_ENABLED = True
    except Exception:
        UART_IRQ_ENABLED = False

# ==================================================
# 摄像头
# ==================================================
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
detect_net = tf.load(
    "yolo3_v2.tflite",
    load_to_fb=True
)

cartoon_net = tf.load(
    "model7.0.tflite",
    load_to_fb=True
)

num_net = tf.load(
    "newnum.tflite",
    load_to_fb=True
)

cartoon_labels = [
    x.rstrip()
    for x in open("/sd/characters.txt")
]

num_labels = [
    x.rstrip()
    for x in open("/sd/numbers.txt")
]

# ==================================================
# 参数
# ==================================================
FRAME_COUNT = 2
YOLO1_THRESHOLD = 0.50
YOLO2_THRESHOLD = 0.50
CLASSIFY_THRESHOLD = 0.70
MIN_VOTES = 2

# 为了先尽量保持和稳定DEBUG版一致
DRAW_MANUAL_ROI = False
DRAW_YOLO1 = False
DRAW_YOLO2 = True

# ==================================================
# ROI
#
#             L2      C2      R2
#
#                     C1
#
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
# 有效范围1~93
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

    if left > 2:
        return None

    if center > 4:
        return None

    if right > 2:
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
# 保持稳定DEBUG版结构
# ==================================================
def yolo_detect(crop, threshold):
    best_rect = None
    best_label = None
    best_score = 0.0

    detections = tf.detect(
        detect_net,
        crop
    )

    for obj in detections:
        try:
            x1, y1, x2, y2, label, score = obj

        except Exception:
            continue

        if score < threshold:
            continue

        img_w = crop.width()
        img_h = crop.height()

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
            best_label = label
            best_rect = (
                x1,
                y1,
                w,
                h
            )

    return (
        best_rect,
        best_label,
        best_score
    )

# ==================================================
# 分类
# ==================================================
def classify_target(target, mode):
    net = (
        cartoon_net
        if mode == 1
        else num_net
    )

    labels = (
        cartoon_labels
        if mode == 1
        else num_labels
    )

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
        pred = list(
            obj.output()
        )

        if not pred:
            continue

        index = pred.index(
            max(pred)
        )

        prob = pred[index]

        if prob > best_prob:
            best_prob = prob
            best_id = get_label_id(
                labels,
                index
            )
            best_name = labels[index]

    return (
        best_id,
        best_name,
        best_prob
    )

# ==================================================
# 单个ROI完整识别
#
# 保持稳定DEBUG版的资源生命周期：
#
# 原图
# ↓
# 手动ROI
# ↓
# YOLO1
# ↓
# YOLO1 Crop
# ↓
# YOLO2
# ↓
# YOLO2 Crop
# ↓
# 分类
# ↓
# 最后统一释放
# ==================================================
def recognize_roi(img, roi, mode):
    roi_x, roi_y, roi_w, roi_h = roi

    if DRAW_MANUAL_ROI:
        img.draw_rectangle(
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            color=(0, 0, 255),
            thickness=1
        )

    # ==============================================
    # 手动ROI
    # ==============================================
    manual_crop = img.copy(
        roi=roi,
        copy_to_fb=False
    )

    # ==============================================
    # YOLO1
    # ==============================================
    yolo1_rect, yolo1_label, yolo1_score = yolo_detect(
        manual_crop,
        YOLO1_THRESHOLD
    )

    if yolo1_rect is None:
        del manual_crop

        return (
            None,
            None,
            0.0,
            None,
            0.0,
            None,
            0.0
        )

    x1, y1, w1, h1 = yolo1_rect

    global_x1 = roi_x + x1
    global_y1 = roi_y + y1

    if DRAW_YOLO1:
        img.draw_rectangle(
            global_x1,
            global_y1,
            w1,
            h1,
            color=(255, 0, 0),
            thickness=2
        )

        img.draw_string(
            global_x1,
            max(
                0,
                global_y1 - 12
            ),
            "Y1 %.2f" % yolo1_score,
            color=(255, 0, 0)
        )

    # ==============================================
    # YOLO1 Crop
    # ==============================================
    yolo1_crop = manual_crop.copy(
        roi=yolo1_rect,
        copy_to_fb=False
    )

    # ==============================================
    # YOLO2
    # ==============================================
    yolo2_rect, yolo2_label, yolo2_score = yolo_detect(
        yolo1_crop,
        YOLO2_THRESHOLD
    )

    if yolo2_rect is None:
        del yolo1_crop
        del manual_crop

        return (
            None,
            None,
            0.0,
            yolo1_label,
            yolo1_score,
            None,
            0.0
        )

    x2, y2, w2, h2 = yolo2_rect

    global_x2 = (
        roi_x +
        x1 +
        x2
    )

    global_y2 = (
        roi_y +
        y1 +
        y2
    )

    if DRAW_YOLO2:
        img.draw_rectangle(
            global_x2,
            global_y2,
            w2,
            h2,
            color=(0, 255, 0),
            thickness=2
        )

        img.draw_string(
            global_x2,
            max(
                0,
                global_y2 - 12
            ),
            "Y2 %.2f" % yolo2_score,
            color=(0, 255, 0)
        )

    # ==============================================
    # YOLO2 Crop
    # ==============================================
    final_target = yolo1_crop.copy(
        roi=yolo2_rect,
        copy_to_fb=False
    )

    # ==============================================
    # 分类
    # ==============================================
    label_id, label_name, class_prob = classify_target(
        final_target,
        mode
    )

    # ==============================================
    # 保持稳定DEBUG版：
    # 最后统一释放
    # ==============================================
    del final_target
    del yolo1_crop
    del manual_crop

    if class_prob < CLASSIFY_THRESHOLD:
        return (
            None,
            None,
            class_prob,
            yolo1_label,
            yolo1_score,
            yolo2_label,
            yolo2_score
        )

    return (
        label_id,
        label_name,
        class_prob,
        yolo1_label,
        yolo1_score,
        yolo2_label,
        yolo2_score
    )

# ==================================================
# 投票
# ==================================================
def vote_result(results):
    stats = {}

    for label_id, label_name, prob in results:
        if label_id is None:
            continue

        if label_id not in stats:
            stats[label_id] = [
                0,
                0.0,
                label_name
            ]

        stats[label_id][0] += 1

        if prob > stats[label_id][1]:
            stats[label_id][1] = prob
            stats[label_id][2] = label_name

    if not stats:
        return (
            None,
            None,
            0.0,
            0
        )

    best_id = None
    best_count = -1
    best_prob = -1.0

    for label_id, value in stats.items():
        count = value[0]
        prob = value[1]

        if (
            count > best_count or
            (
                count == best_count and
                prob > best_prob
            )
        ):
            best_id = label_id
            best_count = count
            best_prob = prob

    value = stats[best_id]

    return (
        best_id,
        value[2],
        value[1],
        value[0]
    )

# ==================================================
# 执行一条完整识别命令
# ==================================================
def process_command(cmd, tasks):
    all_results = [
        [],
        [],
        []
    ]

    start = time.ticks_ms()

    # ==============================================
    # 两帧
    # ==============================================
    for frame in range(FRAME_COUNT):
        img = sensor.snapshot().rotation_corr(
            z_rotation=180,
            zoom=1.0
        )

        # ==========================================
        # L / C / R
        # ==========================================
        for slot in range(3):
            task = tasks[slot]

            if task is None:
                continue

            roi, mode = task

            t0 = time.ticks_ms()

            result = recognize_roi(
                img,
                roi,
                mode
            )

            label_id = result[0]
            label_name = result[1]
            class_prob = result[2]

            yolo1_label = result[3]
            yolo1_score = result[4]
            yolo2_label = result[5]
            yolo2_score = result[6]

            all_results[slot].append(
                (
                    label_id,
                    label_name,
                    class_prob
                )
            )

            if DEBUG_DETAIL:
                slot_name = (
                    "L"
                    if slot == 0
                    else (
                        "C"
                        if slot == 1
                        else "R"
                    )
                )

                if yolo1_label is None:
                    print(
                        "frame=%d %s "
                        "YOLO1=None "
                        "time=%dms" %
                        (
                            frame,
                            slot_name,
                            time.ticks_diff(
                                time.ticks_ms(),
                                t0
                            )
                        )
                    )

                elif yolo2_label is None:
                    print(
                        "frame=%d %s "
                        "Y1=%.3f "
                        "YOLO2=None "
                        "time=%dms" %
                        (
                            frame,
                            slot_name,
                            yolo1_score,
                            time.ticks_diff(
                                time.ticks_ms(),
                                t0
                            )
                        )
                    )

                else:
                    print(
                        "frame=%d %s "
                        "Y1=%.3f "
                        "Y2=%.3f "
                        "class=%s "
                        "prob=%.3f "
                        "time=%dms" %
                        (
                            frame,
                            slot_name,
                            yolo1_score,
                            yolo2_score,
                            str(label_id),
                            class_prob,
                            time.ticks_diff(
                                time.ticks_ms(),
                                t0
                            )
                        )
                    )

        del img
        gc.collect()

    # ==============================================
    # 投票
    # ==============================================
    outputs = [
        None,
        None,
        None
    ]

    names = [
        None,
        None,
        None
    ]

    probs = [
        0.0,
        0.0,
        0.0
    ]

    votes = [
        0,
        0,
        0
    ]

    for slot in range(3):
        if tasks[slot] is None:
            continue

        outputs[slot], names[slot], probs[slot], votes[slot] = vote_result(
            all_results[slot]
        )

        if votes[slot] < MIN_VOTES:
            outputs[slot] = None

    # ==============================================
    # UART发送字符串
    #
    # 只发送参与识别的位置
    #
    # L/C/R = 3/5/7 -> &357%
    # 只有C=5       -> &5%
    # 失败           -> 0
    # ==============================================
    send_str = ""

    for slot in range(3):
        if tasks[slot] is None:
            continue

        value = outputs[slot]

        if value is None:
            value = 0

        send_str += str(value)

    if DEBUG_UART:
        print(
            "RESULT cmd=%d "
            "L=%s C=%s R=%s "
            "vote=%d/%d,%d/%d,%d/%d "
            "time=%dms" %
            (
                cmd,
                str(outputs[0]),
                str(outputs[1]),
                str(outputs[2]),
                votes[0],
                FRAME_COUNT,
                votes[1],
                FRAME_COUNT,
                votes[2],
                FRAME_COUNT,
                time.ticks_diff(
                    time.ticks_ms(),
                    start
                )
            )
        )

    # ==============================================
    # UART TX
    # ==============================================
    if UART_TX_ENABLE:
        tx_data = (
            "&" +
            send_str +
            "%"
        ).encode()

        if DEBUG_UART:
            print(
                "TX:",
                tx_data
            )

        uart.write(tx_data)

        # 尝试等待TX真正发送完成
        try:
            uart.flush()
        except Exception:
            time.sleep_ms(5)

        if DEBUG_UART:
            print("TX DONE")

    gc.collect()

# ==================================================
# UART读取
#
# IRQ只是提醒：
# "有数据来了"
#
# 真正uart.read一定放在主循环
# ==================================================
def read_uart_cmd():
    global uart_irq_flag

    # IRQ没有发生且RX buffer为空
    if (
        uart_irq_flag == 0 and
        uart.any() <= 0
    ):
        return None

    # 清除IRQ标志
    uart_irq_flag = 0

    latest_cmd = None

    # ==============================================
    # 把当前RX buffer全部读完
    # ==============================================
    while True:
        n = uart.any()

        if n <= 0:
            break

        data = uart.read(n)

        if not data:
            break

        for cmd in data:
            tasks = decode_cmd(cmd)

            if tasks is None:
                if DEBUG_UART:
                    print(
                        "RX INVALID:",
                        cmd
                    )

                continue

            # 只保留最新有效命令
            latest_cmd = cmd

    return latest_cmd

# ==================================================
# 启动前清空可能存在的旧串口数据
# ==================================================
try:
    while uart.any() > 0:
        uart.read(
            uart.any()
        )
except Exception:
    pass

uart_irq_flag = 0

# ==================================================
# 启动信息
# ==================================================
print("==============================")
print("UART IRQ YOLO")
print("UART12 = 115200")
print("FRAME_COUNT =", FRAME_COUNT)
print("IRQ ENABLE =", UART_IRQ_ENABLED)
print("UART TX =", UART_TX_ENABLE)
print("==============================")

# ==================================================
# 主循环
#
# 空闲：
# 只检查UART
#
# 收到：
# 完整识别
# ↓
# 返回结果
# ↓
# 再次等待UART
# ==================================================
while True:
    cmd = read_uart_cmd()

    if cmd is None:
        time.sleep_ms(1)
        continue

    tasks = decode_cmd(cmd)

    if tasks is None:
        continue

    # ==============================================
    # 如果第二次命令能够来到这里，
    # 说明UART RX已经成功
    # ==============================================
    if DEBUG_UART:
        print("")
        print("==============================")
        print("RX CMD:", cmd)
        print(
            "BEFORE CMD irq=%d any=%d" %
            (
                uart_irq_flag,
                uart.any()
            )
        )

    gc.collect()

    # ==============================================
    # 正式识别
    # ==============================================
    process_command(
        cmd,
        tasks
    )

    gc.collect()

    # ==============================================
    # 如果看到CMD DONE，
    # 说明整轮识别+TX都执行完成
    # ==============================================
    if DEBUG_UART:
        print(
            "CMD DONE:",
            cmd
        )

        print(
            "AFTER CMD irq=%d any=%d" %
            (
                uart_irq_flag,
                uart.any()
            )
        )

        print("==============================")
