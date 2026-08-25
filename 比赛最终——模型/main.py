import sensor,time,tf,gc
from machine import UART
uart=UART(12,baudrate=115200)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_brightness(200)
sensor.set_windowing(240,240)
sensor.skip_frames(time=2000)
clock=time.clock()

detect_net_path="yolo3_v2.tflite"
detect_net=tf.load(detect_net_path,load_to_fb=True)

cartoon_net_path="model7.0.tflite"
cartoon_labels=[line.rstrip() for line in open("/sd/characters.txt")]
cartoon_net=tf.load(cartoon_net_path,load_to_fb=True)

num_net_path="newnum.tflite"
num_labels=[line.rstrip() for line in open("/sd/numbers.txt")]
num_net=tf.load(num_net_path,load_to_fb=True)

# mode为1则开启debug
debugmode=0
# task为0是目标识别
debugtask=0

while(not debugmode):
    img=sensor.snapshot().rotation_corr(z_rotation=180,zoom=1.0)
    uart_num=uart.any()
    if(uart_num):
        uart_str=uart.read(uart_num).decode()
        result_labels=[]
        result_probs=[]
        for i in range(3):
            img=sensor.snapshot().rotation_corr(z_rotation=180,zoom=1.0)
            img_square=img.copy(0.75,1)
            rects=tf.detect(detect_net,img_square)
            if rects:
                for obj in rects:
                    x1,y1,x2,y2,label,scores=obj
                    if(scores>0.60):
                        w=x2-x1
                        h=y2-y1
                        x1=int(x1*img.width())
                        y1=int(y1*img.height())
                        w=int(w*img.width())
                        h=int(h*img.height())
                        img1=img.copy(roi=(x1,y1,w,h),copy_to_fb=False)
                        if "G" in uart_str:
                            for obj in tf.classify(cartoon_net,img1,min_scale=1,scale_mul=0.8,x_overlap=0.5,y_overlap=0.5):
                                predictions=list(obj.output())
                                max_index=predictions.index(max(predictions))
                                result_labels.append(cartoon_labels[max_index])
                                result_probs.append(predictions[max_index])
                                break
                        elif "N" in uart_str:
                            for obj in tf.classify(num_net,img,min_scale=1,scale_mul=0.8,x_overlap=0.5,y_overlap=0.5):
                                predictions=list(obj.output())
                                max_index=predictions.index(max(predictions))
                                result_labels.append(num_labels[max_index])
                                result_probs.append(predictions[max_index])
                                break
                        break
            else:
                if "G" in uart_str:
                    for obj in tf.classify(cartoon_net,img,min_scale=1,scale_mul=0.8,x_overlap=0.5,y_overlap=0.5):
                        predictions=list(obj.output())
                        max_index=predictions.index(max(predictions))
                        result_labels.append(cartoon_labels[max_index])
                        result_probs.append(predictions[max_index])
                        break
                elif "N" in uart_str:
                    for obj in tf.classify(num_net,img,min_scale=1,scale_mul=0.8,x_overlap=0.5,y_overlap=0.5):
                        predictions=list(obj.output())
                        max_index=predictions.index(max(predictions))
                        result_labels.append(num_labels[max_index])
                        result_probs.append(predictions[max_index])
                        break
                break


            gc.collect()
        if result_labels:
            max_count=0
            final_label=result_labels[0]
            for label in result_labels:
                count=result_labels.count(label)
                if count>max_count:
                    max_count=count
                    final_label=label
            if max_count==1:
                final_label=result_labels[result_probs.index(max(result_probs))]
            uart.write("&"+final_label+"%")
        else:
            uart.write("&3%")
        uart.read()
        gc.collect()

while(debugmode):
    img=sensor.snapshot().rotation_corr(z_rotation=180,zoom=1.0)
    img_square=img.copy(0.75,1)
    rects=tf.detect(detect_net,img_square)
    if rects:
        for obj in rects:
            x1,y1,x2,y2,label,scores=obj
            if(scores>0.70):
                w=x2-x1
                h=y2-y1
                x1=int(x1*img.width())
                y1=int(y1*img.height())
                w=int(w*img.width())
                h=int(h*img.height())
                img.draw_rectangle(x1,y1,w,h,(255,255,255),1)
                img1=img.copy(roi=(x1,y1,w,h),copy_to_fb=False)
                if debugtask==0:
                    for obj in tf.classify(cartoon_net,img1,min_scale=1,scale_mul=0.8,x_overlap=0.5,y_overlap=0.5):
                        predictions=list(obj.output())
                        max_index=predictions.index(max(predictions))
                        max_label=cartoon_labels[max_index]
                        max_prob=predictions[max_index]
                        print("%s:%f"%(max_label,max_prob))
                        break
                else:
                    for obj in tf.classify(num_net,img1,min_scale=1,scale_mul=0.8,x_overlap=0.5,y_overlap=0.5):
                        predictions=list(obj.output())
                        max_index=predictions.index(max(predictions))
                        max_label=num_labels[max_index]
                        max_prob=predictions[max_index]
                        print("%s:%f"%(max_label,max_prob))
                        break
                break
    gc.collect()

# while (debugmode):
#     clock.tick()

#     img = sensor.snapshot().rotation_corr(
#         z_rotation=180,
#         zoom=1.0
#     )

#     left_img = img.copy(roi=(160,0,320,240))

#     rects = tf.detect(detect_net, left_img)

#     count = 0

#     for obj in rects:

#         x1, y1, x2, y2, label, score = obj

#         print("raw:", obj)

#         if score > 0.60:

#             count += 1

#             x1 = int(x1 * left_img.width())
#             y1 = int(y1 * left_img.height())
#             x2 = int(x2 * left_img.width())
#             y2 = int(y2 * left_img.height())

#             w = x2 - x1
#             h = y2 - y1

#             cx = x1 + w // 2
#             cy = y1 + h // 2

#             img.draw_rectangle(
#                 x1, y1, w, h,
#                 (255,255,255),
#                 1
#             )

#             print(
#                 "target:",
#                 count,
#                 "cx:", cx,
#                 "cy:", cy,
#                 "score:", score
#             )

#     print("detect count:", count)

#     gc.collect()
