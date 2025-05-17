import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path="pic"
images= []
classNames =[]
myList =os.listdir(path)
print(myList)
for cl in myList:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    #splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
print(len(images))
print(classNames)

#step2 encodeing
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #BGR được chuyển đổi sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("encodeing complete")
print(len(encodeListKnown))

def thamdu(name):
    with open('thamdu.csv', 'r+') as f:
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',') # tách theo dấu ,
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now() # trả về 2021-12-18 16:43:30.709791
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



# khởi động webcam, mã hóa hình ảnh webcam
cap =cv2.VideoCapture("videocheck.mp4")

while True:
    ret, frame= cap.read()
    framS= cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    #xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace,faceLoc in zip(encodecurFrame,facecurFrame):  # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        # tìm faceDis thấp nhất sẽ chính nhận dạng được người cần tìm
        matchIndex = np.argmin(faceDis) #đẩy về index của faceDis nhỏ nhất
        print(matchIndex)


        if faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            thamdu(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Nhận diện', frame)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ