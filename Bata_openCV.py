# 匯入必要套件
import argparse
import ntpath
import os
import cv2
import imutils
# python Bata_openCV.py -p eye
#匯入模型
detectors = {
    "eye": os.path.sep.join([ntpath.dirname(cv2.__file__), 'data', 'haarcascade_eye.xml']),
    "face": os.path.sep.join([ntpath.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'])
}

# 定義人臉偵測函數方便重複使用
def detect(gray, part="face"):
    # 初始化Haar cascades函數
    detector = cv2.CascadeClassifier(detectors[part])
    # 根據選擇的模型偵測
    rects = detector.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=4, minSize=(15, 15),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
#ScaleFactor：每次搜尋方塊減少的比例 minNeighbers：每個目標至少檢測到幾次以上，才可被認定是真數據。 minSize：設定數據搜尋的最小尺寸 ，如 minSize=(40,40)
    return rects

# 初始化arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--part", type=str, choices=["eye", "face"], default="face", help="detect which part of face")
args = vars(ap.parse_args())

# 取得當前的img，變更比例為，並且轉成灰階圖片
img = input("圖片的名字")
img = cv2.imread(img)

# img = cv2.imread('000.jpeg')

gray = imutils.resize(img, width=500)#放大圖片 越大越細
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)#把圖變灰階圖片

# 呼叫偵測函數，取得結果
rects = detect(gray, args["part"])

# 繪出偵測結果 (記得將偵測的座標轉回原本的img大小)
ratio = img.shape[1] / gray.shape[1]
for rect in rects:
    rect = rect * ratio
    (x, y, w, h) = rect.astype("int")#把rect numpy轉int
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 顯示影像


print("找到了 {0} 張臉.".format(len(rects)))
cv2.namedWindow('img', cv2.WINDOW_NORMAL)  #正常視窗大小
cv2.imshow('img', img)                     #秀出圖片
cv2.imwrite( "result.jpg", img )           #保存圖片
cv2.waitKey(0)                             #等待按下任一按鍵
cv2.destroyAllWindows()   