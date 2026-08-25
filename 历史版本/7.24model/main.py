import sensor,time,tf,gc
from machine import UART
uart = UART(12, baudrate=115200)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)# we run out of memory if the resolution is much bigger...
sensor.set_windowing(240,240)
# sensor.set_brightness(2000)
sensor.skip_frames(time = 2000)
# sensor.set_auto_gain(False)  # must turn this off to prevent image washout...
# sensor.set_auto_whitebal(True,(0,0x80,0))  # must turn this off to prevent image washout...
clock = time.clock()

# target_thresholds = (17, 88, 64, 107, -86, -29)
# box_thresholds = (17, 95, -42, 34, 21, 93)

detect_net_path = "yolo3_v2.tflite"
detect_net = tf.load(detect_net_path,load_to_fb=True)

cartoon_net_path = "model7.0.tflite"
cartoon_labels = [line.rstrip() for line in open("/sd/characters.txt")]
cartoon_net = tf.load(cartoon_net_path, load_to_fb=True)

task_1_switch = 0

num_net_path = "newnum.tflite"
num_labels = [line.rstrip() for line in open("/sd/numbers.txt")]
num_net = tf.load(num_net_path, load_to_fb=True)


# mode为1则开启debug
debugmode = 0
# task为0是目标识别
debugtask = 0

# def is_fake_object(img1):

#     area = img1.width() * img1.height()
#     blobs = img1.find_blobs(
#         [target_thresholds, box_thresholds],
#         pixels_threshold=1000,
#         area_threshold=1000,
#         merge=True
#     )

#     if not blobs:
#         return False

#     max_blob = max(blobs, key=lambda b: b.pixels())
#     ratio = max_blob.pixels() / area

#     # print("fake check ratio:", ratio)

#     if ratio > 0.75:
#         return True
#     else:
#         return False

while(not debugmode):
    img = sensor.snapshot().rotation_corr(z_rotation=180,zoom = 1.0)
    uart_num = uart.any()       # 获取当前串口数据数量
    if(uart_num):
        img = sensor.snapshot().rotation_corr(z_rotation=180,zoom = 1.0)
        uart_str = uart.read(uart_num).decode() # 读取串口数据
        img_square = img.copy(0.75, 1)
        rects = tf.detect(detect_net,img_square)
        if rects:
            for obj in rects:
                x1,y1,x2,y2,label,scores = obj
                if(scores>0.70):
                    # print(obj)
                    w = x2- x1
                    h = y2 - y1
                    x1 = int((x1)*img.width())
                    y1 = int(y1*img.height())
                    w = int(w*img.width())
                    h = int(h*img.height())
                    # img.draw_rectangle((x1,y1,w,h),thickness=2)
                    img1 = img.copy(roi = (x1,y1,w,h),copy_to_fb=False)
                    # if task_1_switch == 0 and is_fake_object(img1):
                    #     print("error")

                    # else:
                    #     task_1_switch = 1
                    if "G" in uart_str:
                        for obj in tf.classify(cartoon_net, img1, min_scale=1, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                            predictions = list(obj.output())
                            max_index = predictions.index(max(predictions))
                            max_label = cartoon_labels[max_index]
                            max_prob = predictions[max_index]
                            uart.write("&"+max_label+"%")
                            break

                    elif "N" in uart_str:
                        for obj in tf.classify(num_net, img1, min_scale=1, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                            predictions = list(obj.output())
                            max_index = predictions.index(max(predictions))
                            max_label = num_labels[max_index]
                            max_prob = predictions[max_index]
                            uart.write("&"+max_label+"%")
                            break

                    gc.collect()
                    uart.read()
                    break

                else:
                    uart.write("posibility error")
                    break
        else:
            uart.write("detect error")
            continue


while(not debugmode):
    img = sensor.snapshot().rotation_corr(z_rotation=180,zoom = 1.0)
    uart_num = uart.any()       # 获取当前串口数据数量
    if(uart_num):
        img = sensor.snapshot().rotation_corr(z_rotation=180,zoom = 1.0)
        uart_str = uart.read(uart_num).decode() # 读取串口数据
        img_square = img.copy(0.75, 1)
        rects = tf.detect(detect_net,img_square)
        if rects:
            for obj in rects:
                x1,y1,x2,y2,label,scores = obj
                if(scores>0.70):
                    # print(obj)
                    w = x2- x1
                    h = y2 - y1
                    x1 = int((x1)*img.width())
                    y1 = int(y1*img.height())
                    w = int(w*img.width())
                    h = int(h*img.height())
                    # img.draw_rectangle((x1,y1,w,h),thickness=2)
                    img1 = img.copy(roi = (x1,y1,w,h),copy_to_fb=False)
                    # if task_1_switch == 0 and is_fake_object(img1):
                    #     print("error")

                    # else:
                    #     task_1_switch = 1
                    if "G" in uart_str:
                        for obj in tf.classify(cartoon_net, img1, min_scale=1, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                            predictions = list(obj.output())
                            max_index = predictions.index(max(predictions))
                            max_label = cartoon_labels[max_index]
                            max_prob = predictions[max_index]
                            uart.write("&"+max_label+"%")
                            break

                    elif "N" in uart_str:
                        for obj in tf.classify(num_net, img1, min_scale=1, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                            predictions = list(obj.output())
                            max_index = predictions.index(max(predictions))
                            max_label = num_labels[max_index]
                            max_prob = predictions[max_index]
                            uart.write("&"+max_label+"%")
                            break
                    gc.collect()
                    uart.read()
                    break
                else:
                    # uart.write("possibility error")
                    uart.write("&"+"3"+"%")

                    break
        else:
            uart.write("&"+"3"+"%")
            # uart.write("detect error")
            continue


while(debugmode):
    img = sensor.snapshot().rotation_corr(z_rotation=180,zoom = 1.0)
    img_square = img.copy(0.75, 1)
    rects = tf.detect(detect_net,img_square)
    if rects:
        for obj in tf.detect(detect_net,img_square):
            x1,y1,x2,y2,label,scores = obj
            if(scores>0.70):
                # print(obj)
                w = x2- x1
                h = y2 - y1
                x1 = int((x1)*img.width())
                y1 = int(y1*img.height())
                w = int(w*img.width())
                h = int(h*img.height())
                # img.draw_rectangle((x1,y1,w,h),thickness=2)
                img1 = img.copy(roi = (x1,y1,w,h),copy_to_fb=False)

                if debugtask == 0:
                    for obj in tf.classify(cartoon_net, img1, min_scale=1, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                        predictions = list(obj.output())
                        max_index = predictions.index(max(predictions))
                        max_label = cartoon_labels[max_index]
                        max_prob = predictions[max_index]
                        print(f"{max_label}:{max_prob}")
                else:
                    for obj in tf.classify(num_net, img1, min_scale=1, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
                        predictions = list(obj.output())
                        max_index = predictions.index(max(predictions))
                        max_label = num_labels[max_index]
                        max_prob = predictions[max_index]
                        print(f"{max_label}:{max_prob}")

# for obj in tf.classify(cartoon_net,img):
#     sorted_list = sorted(zip(cartoon_labels, obj.output()), key = lambda x: x[1], reverse = True)
    # print("%s = %f" % (sorted_list[0][0], sorted_list[0][1]
