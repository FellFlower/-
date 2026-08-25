import sensor,time,tf,gc

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_framerate(60)
sensor.set_brightness(200)
sensor.set_windowing(320,240)
sensor.skip_frames(time=2000)

net=tf.load("yolo3_v2.tflite",load_to_fb=True)
THRESHOLD=0.60

# 七个ROI
ROI_C1=(32,6,238,223)
ROI_L2=(10,70,101,99)
ROI_C2=(97,68,98,98)
ROI_R2=(183,67,97,98)
ROI_L3=(57,84,67,67)
ROI_C3=(111,83,66,68)
ROI_R3=(165,83,67,68)

# [左,中,右]
# 左：0=不检测 1=L2 2=L3
# 中：0=C1    1=C2 2=C3
# 右：0=不检测 1=R2 2=R3
POS=[0,0,0]

def get_rois(pos):
    left=None if pos[0]==0 else (ROI_L2 if pos[0]==1 else ROI_L3)
    center=ROI_C1 if pos[1]==0 else (ROI_C2 if pos[1]==1 else ROI_C3)
    right=None if pos[2]==0 else (ROI_R2 if pos[2]==1 else ROI_R3)
    return [left,center,right]

def detect_roi(img,roi,name):
    if roi is None:return
    rx,ry,rw,rh=roi
    crop=img.copy(roi=roi,copy_to_fb=False)
    rects=tf.detect(net,crop)

    for obj in rects:
        x1,y1,x2,y2,label,score=obj
        if score<THRESHOLD:continue

        px1=rx+int(x1*rw)
        py1=ry+int(y1*rh)
        px2=rx+int(x2*rw)
        py2=ry+int(y2*rh)

        if px1<0:px1=0
        if py1<0:py1=0
        if px2>319:px2=319
        if py2>239:py2=239

        w=px2-px1
        h=py2-py1
        if w>0 and h>0:
            img.draw_rectangle(px1,py1,w,h,color=(255,255,255),thickness=1)
            print("%s score=%.3f rect=(%d,%d,%d,%d)"%(name,score,px1,py1,w,h))

    del crop

ROIS=get_rois(POS)
NAMES=["LEFT","CENTER","RIGHT"]

while True:
    img=sensor.snapshot().rotation_corr(z_rotation=180,zoom=1.0)
    for i in range(3):
        detect_roi(img,ROIS[i],NAMES[i])
    gc.collect()
    # print(net.input_width())
    # print(net.input_height())
