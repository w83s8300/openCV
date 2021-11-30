import ntpath
import sys


# python face_recognition/dlib_with_opencv_realtime.py -i dataset/caltech_faces -e embeddings.pickle

# resolve module import error in PyCharm
sys.path.append(ntpath.dirname(ntpath.dirname(ntpath.abspath(__file__))))

import argparse
import os
import pickle
import time

import cv2
import face_recognition

from imutils.video import WebcamVideoStream

from dataset.load_dataset import load_images
from openCV_HOG_SVM import detect as hog_detect
from openCV_Dlib_MMOD import detect as mmod_detect
from openCV_DNN import detect as detect_dnns


def main():
    # 初始化arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="the input dataset path")
    ap.add_argument("-e", "--embeddings-file", type=str, required=True,
                    help="the path to serialized db of facial embeddings")
    ap.add_argument("-d", "--detection-method", type=str, default="mmod", choices=["hog", "mmod"],
                    help="the detection method to use")
    args = vars(ap.parse_args())

    print("[INFO] loading dataset....")
    (faces, names) = load_images(args["input"], min_size=10)
    # 由於Dlib處理圖片不同於OpenCV的BGR順序，需要先轉換成RGB順序
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
    print(f"[INFO] {len(faces)} images in dataset")

    # 初始化結果
    known_embeddings = []
    known_names = []

    # 建立我們的人臉embeddings資料庫
    data = {}
    print("[INFO] serializing embeddings...")
    if os.path.exists(args["embeddings_file"]):
        with open(args["embeddings_file"], "rb") as f:
            data = pickle.load(f)
    else:
        start = time.time()
        for (img, name) in zip(faces, names):
            # 偵測人臉位置
            if args["detection_method"] == "mmod":
                rects = detect_dnns(img)
            else:
                rects = hog_detect(img)
            # 將我們偵測的結果(x, y, w, h)轉為face_recognition使用的box格式: (top, right, bottom, left)
            boxes = [(rect[1], rect[1] + rect[3], rect[0] + rect[2], rect[0]) for rect in rects]
            embeddings = face_recognition.face_encodings(img, boxes)
            for embedding in embeddings:
                known_embeddings.append(embedding)
                known_names.append(name)

        print("[INFO] saving embeddings to file...")
        data = {"embeddings": known_embeddings, "names": known_names}
        with open(args["embeddings_file"], "wb") as f:
            pickle.dump(data, f)
        end = time.time()
        print(f"[INFO] serializing embeddings done, tooks {round(end - start, 3)} seconds")

    # 啟動WebCam
    vs = WebcamVideoStream().start()
    time.sleep(2.0)
    fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))

    #讀取比對照片



    while True:
        frame = vs.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img, model="cnn" if args["detection_method"] == "mmod" else "hog")
        embeddings = face_recognition.face_encodings(img, boxes)

        # 辨識結果
        names = []
        for embedding in embeddings:
            matches = face_recognition.compare_faces(data["embeddings"], embedding)
            name = "unknown"
            # matches是一個包含True/False值的list，會比對所有資料庫中的人臉embeddings
            if True in matches:
                # 判斷哪一個人有最多matches
                matchedIdexs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdexs:
                    name = data["names"][i]
                    
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)
            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 8 if top - 8 > 8 else top + 8
            cv2.putText(img, f"{name}", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Result", img)
        print("名字是:",name)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break


if __name__ == '__main__':
    main()