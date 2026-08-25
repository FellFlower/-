import sensor, time, tf, gc

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

# 第一次YOLO阈值
YOLO1_THRESHOLD = 0.5

# 第二次YOLO阈值
YOLO2_THRESHOLD = 0.5

# 分类阈值
CLASSIFY_THRESHOLD = 0.70

# 最少有效投票数
MIN_VOTES = 2

# 显示开关
DRAW_MANUAL_ROI = False
DRAW_YOLO1 = False
DRAW_YOLO2 = True

# 是否输出每一帧详细信息
DEBUG_DETAIL = False

# ==================================================
# DEBUG输入
#
# DEBUG_INPUT = [左, 中, 右]
#
# 左：
# 0 = L2人物
# 1 = L2数字
#
# 中：
# 0 = C1人物
# 1 = C1数字
# 2 = C2人物
# 3 = C2数字
#
# 右：
# 0 = R2人物
# 1 = R2数字
# ==================================================
DEBUG_INPUT = [0, 2, 0]

# ==================================================
# 手动ROI
#
#             L2      C2      R2
#
#                     C1
#
#                     车
# ==================================================
ROI_C1 = (36, 1, 224, 218)
ROI_L2 = (0,22,136,148)
ROI_C2 = (74,22,137,149)
ROI_R2 = (166,21,135,150)


def decode_debug_input(data):
    if len(data) != 3:
        return None

    left, center, right = data

    if left not in (0, 1):
        return None
    if center not in (0, 1, 2, 3):
        return None
    if right not in (0, 1):
        return None

    left_task = (ROI_L2, 1 if left == 0 else 2)

    if center == 0:
        center_task = (ROI_C1, 1)
    elif center == 1:
        center_task = (ROI_C1, 2)
    elif center == 2:
        center_task = (ROI_C2, 1)
    else:
        center_task = (ROI_C2, 2)

    right_task = (ROI_R2, 1 if right == 0 else 2)

    return [left_task, center_task, right_task]


def get_label_id(labels, index):
    try:
        v = int(labels[index])
        if 0 <= v <= 9:
            return v
    except Exception:
        pass

    return index if index <= 9 else 0


# ==================================================
# 通用YOLO检测函数
#
# threshold：
# 可分别用于第一次和第二次YOLO
#
# 返回：
# rect = (x,y,w,h)
# label
# score
# ==================================================
def yolo_detect(crop, threshold):
    best_rect = None
    best_label = None
    best_score = 0.0

    detections = tf.detect(detect_net, crop)

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
            best_rect = (x1, y1, w, h)

    return best_rect, best_label, best_score


# ==================================================
# 分类
# mode:
# 1 = 人物
# 2 = 数字
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
# 完整二次YOLO识别
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
# ==================================================
def recognize_roi(img, roi, mode):
    roi_x, roi_y, roi_w, roi_h = roi

    # ------------------------------------------------
    # 手动ROI显示
    # ------------------------------------------------
    if DRAW_MANUAL_ROI:
        img.draw_rectangle(
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            color=(0, 0, 255),
            thickness=1
        )

    # ------------------------------------------------
    # 裁出手动ROI
    # ------------------------------------------------
    manual_crop = img.copy(
        roi=roi,
        copy_to_fb=False
    )

    # =================================================
    # 第一次YOLO
    # =================================================
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

    # 第一次YOLO在原图上的坐标
    global_x1 = roi_x + x1
    global_y1 = roi_y + y1

    if DRAW_YOLO1:
        img.draw_rectangle(
            global_x1,
            global_y1,
            h1,
            h1,
            color=(255, 0, 0),
            thickness=2
        )

        img.draw_string(
            global_x1,
            max(0, global_y1 - 12),
            "Y1 %.2f" % yolo1_score,
            color=(255, 0, 0)
        )

    # ------------------------------------------------
    # 裁第一次YOLO框
    # ------------------------------------------------
    yolo1_crop = manual_crop.copy(
        roi=yolo1_rect,
        copy_to_fb=False
    )

    # =================================================
    # 第二次YOLO
    # =================================================
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

    # ------------------------------------------------
    # 第二次YOLO坐标换算
    #
    # 第二次坐标是相对于第一次YOLO Crop
    #
    # 所以原图位置：
    # ROI偏移 + YOLO1偏移 + YOLO2偏移
    # ------------------------------------------------
    global_x2 = roi_x + x1 + x2
    global_y2 = roi_y + y1 + y2

    if DRAW_YOLO2:
        img.draw_rectangle(
            global_x2,
            global_y2,
            h2,
            h2,
            color=(0, 255, 0),
            thickness=2
        )

        img.draw_string(
            global_x2,
            max(0, global_y2 - 12),
            "Y2 %.2f" % yolo2_score,
            color=(0, 255, 0)
        )

    # ------------------------------------------------
    # 裁出第二次YOLO最终目标
    # ------------------------------------------------
    final_target = yolo1_crop.copy(
        roi=yolo2_rect,
        copy_to_fb=False
    )

    # =================================================
    # 分类
    # =================================================
    label_id, label_name, class_prob = classify_target(
        final_target,
        mode
    )

    del final_target
    del yolo1_crop
    del manual_crop

    # ------------------------------------------------
    # 分类置信度过滤
    # ------------------------------------------------
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
# 3帧投票
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
        return None, None, 0.0, 0

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

    value = stats[best_id]

    return (
        best_id,
        value[2],
        value[1],
        value[0]
    )


# ==================================================
# DEBUG初始化
# ==================================================
tasks = decode_debug_input(DEBUG_INPUT)

if tasks is None:
    print("DEBUG_INPUT ERROR")

    while True:
        pass

print("==============================")
print("DEBUG INPUT:", DEBUG_INPUT)

print(
    "L =",
    "L2人物" if DEBUG_INPUT[0] == 0 else "L2数字"
)

if DEBUG_INPUT[1] == 0:
    print("C = C1人物")
elif DEBUG_INPUT[1] == 1:
    print("C = C1数字")
elif DEBUG_INPUT[1] == 2:
    print("C = C2人物")
else:
    print("C = C2数字")

print(
    "R =",
    "R2人物" if DEBUG_INPUT[2] == 0 else "R2数字"
)

print("YOLO1 threshold =", YOLO1_THRESHOLD)
print("YOLO2 threshold =", YOLO2_THRESHOLD)
print("CLASS threshold =", CLASSIFY_THRESHOLD)
print("==============================")


# ==================================================
# 主循环
# ==================================================
while True:
    all_results = [[], [], []]

    start = time.ticks_ms()

    # =================================================
    # 连续采3帧
    # =================================================
    for frame in range(FRAME_COUNT):

        img = sensor.snapshot().rotation_corr(
            z_rotation=180,
            zoom=1.0
        )

        # =============================================
        # 每帧识别L/C/R三个位置
        # =============================================
        for slot in range(3):
            roi, mode = tasks[slot]

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

            # =========================================
            # DEBUG详细输出
            # =========================================
            if DEBUG_DETAIL:

                slot_name = (
                    "L" if slot == 0
                    else ("C" if slot == 1 else "R")
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
                        "class_prob=%.3f "
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

    # =================================================
    # 3帧投票
    # =================================================
    outputs = [None, None, None]
    names = [None, None, None]
    probs = [0.0, 0.0, 0.0]
    votes = [0, 0, 0]

    for i in range(3):
        outputs[i], names[i], probs[i], votes[i] = vote_result(
            all_results[i]
        )

        # ---------------------------------------------
        # 最少投票次数过滤
        # ---------------------------------------------
        if votes[i] < MIN_VOTES:
            outputs[i] = None

    # =================================================
    # 输出
    # =================================================
    print(
        "RESULT "
        "L=%s C=%s R=%s "
        "vote=%d/%d,%d/%d,%d/%d "
        "prob=%.3f,%.3f,%.3f "
        "time=%dms" %
        (
            str(outputs[0]),
            str(outputs[1]),
            str(outputs[2]),

            votes[0],
            FRAME_COUNT,

            votes[1],
            FRAME_COUNT,

            votes[2],
            FRAME_COUNT,

            probs[0],
            probs[1],
            probs[2],

            time.ticks_diff(
                time.ticks_ms(),
                start
            )
        )
    )

    gc.collect()
