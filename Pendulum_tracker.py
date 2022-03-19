from numpy.lib.function_base import angle
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture("150(5).MOV")
ret, img = cap.read()
image = img[:, 475:1445]
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 46, 64])
upper = np.array([255, 255, 255])
mask = cv2.inRange(image, lower, upper)
img_erosion = cv2.erode(mask, (3, 3), iterations=8)
conturs, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, conturs, -1, (0, 255, 0), 3)
for contur in conturs:
    (x1, y1, w1, h1) = cv2.boundingRect(contur)
    if cv2.contourArea(contur) < 150:
        continue
    center = (int((x1 + x1 + w1) / 2), int((y1 + y1 + h1) / 2))
center=list(center)
center[1]=center[1]-18
angle = []
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        img_crp = img[:, 475:1445]
        frame = cv2.cvtColor(img_crp, cv2.COLOR_BGR2GRAY)
    else:
        break
    _, thres = cv2.threshold(frame, 90, 255, cv2.THRESH_BINARY)
    img_erosion = cv2.erode(thres, (3, 3), iterations=8)
    conturs, hierarchy = cv2.findContours(
        img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    for contur in conturs:
        (x, y, w, h) = cv2.boundingRect(contur)
        if cv2.contourArea(contur) < 10000:
            continue
        cv2.rectangle(img_crp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cen_rec = (int((2 * x + w) / 2), int((2 * y + h) / 2))
        cv2.circle(img_crp, cen_rec, 6, (255, 0, 0), -1)
        # delx=center[0]-((2*x + w)/2)
        # dely=center[1]-((2*y + h)/2)

        # if delx < 0 and dely > 0:
        #     sign_array.append(1)
        #     angle.append(h)
        # if delx >= 0 and dely >= 0:
        #     sign_array.append(2)
        #     angle.append(h)
        # if delx < 0 and dely < 0:
        #     sign_array.append(3)
        #     angle.append(h)
        # if delx > 0 and dely < 0:
        #     sign_array.append(4)
        #     angle.append(h)
    t_y=cen_rec[1]-center[0]
    t_x=cen_rec[0]-center[0]
    try:
        thetha=(math.atan(t_y/t_x))*57.2958
        if t_x>0 and t_y>0:
            ang=thetha-90
            text = "1st"
        elif t_x<0 and t_y>0:
            ang=90+thetha
            text = "2nd"
        elif t_x<0 and t_y<0:
            ang=90+thetha
            text = "3rd"
        elif t_x>0 and t_y<0:
            ang=-90+thetha
            text = "4th"

    except ZeroDivisionError:
        if t_y>0:
            ang=90
        elif t_y<0:
            ang=270
        pass
    angle.append(ang)
    cv2.putText(img_crp,str(ang)+" " + text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
    cv2.circle(img_crp, center, 5, (255, 255, 0), -1)
    cv2.imshow("real", img_crp)
    if cv2.waitKey(1) == ord("q"):
        break
    time.sleep(0.1)
print(angle)
cap.release()
cv2.destroyAllWindows()
print(angle)
# plt.plot(angle)
# plt.show()
