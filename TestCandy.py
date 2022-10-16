# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Test1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
from mmcv import Config
from mmdet.apis.inference import inference_detector
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import numpy as np
from solo_v2 import detect_center
from solo_v2 import detect_center_bbox
import time
import socket
from multiprocessing import Process
import CameraUtils
import TCPIP

def createData(coords_list):
    list1 = [elem for coords in coords_list for elem in coords]
    msg = ""
    for i in list1:
        msg += ","
        msg += str(i)
    msg = str(len(coords_list)) + msg
    return msg

def zeroExtend(inputList):
    output = []
    for tup in inputList:
        tup = (*tup, 0)
        output.append(tup)
    return output

def clientUtilities(data):
    PORT = 5050
    # SERVER = "192.168.1.1"
    SERVER = "127.0.0.1"
    ADDR = (SERVER, PORT)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)

    def send(msg):
        HEADER = 64
        FORMAT = 'utf-8'
        message = msg.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        client.send(send_length)
        client.send(message)

    send(createData(zeroExtend(data)))
    client.close()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1105, 578)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QtCore.QRect(680, 480, 93, 28))
        self.loadButton.setObjectName("loadButton")
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(790, 480, 93, 28))
        self.saveButton.setObjectName("saveButton")
        self.runButton = QtWidgets.QPushButton(self.centralwidget)
        self.runButton.setGeometry(QtCore.QRect(900, 480, 93, 28))
        self.runButton.setObjectName("runButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(680, 400, 121, 16))
        self.label.setObjectName("label")
        self.infTimeLabel = QtWidgets.QLabel(self.centralwidget)
        self.infTimeLabel.setGeometry(QtCore.QRect(800, 400, 121, 16))
        self.infTimeLabel.setObjectName("infTimeLabel")
        self.displayLabel = QtWidgets.QLabel(self.centralwidget)
        self.displayLabel.setGeometry(QtCore.QRect(20, 30, 640, 480))
        self.displayLabel.setObjectName("displayLabel")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(690, 90, 321, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1020, 90, 55, 16))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1105, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.thresholdValue = 0.5
        self.time = 0.0
        self.time_start = 0
        self.time_detect = 0

        self.retranslateUi(MainWindow)
        self.loadButton.clicked.connect(self.loadImage)
        self.saveButton.clicked.connect(self.saveImage)
        self.runButton.clicked.connect(self.runInferenceVideo)
        self.horizontalSlider.valueChanged['int'].connect(self.updateConfidence)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def updateConfidence(self, value):
        self.thresholdValue = value/100
        self.label_2.setNum(value)


    def loadImage(self):
        self.filename = QFileDialog.getOpenFileName(directory="C:/Users/LAPTOP/Desktop/Pics")[0]
        self.image = cv2.imread(self.filename)
        self.set_image(self.image)

    def saveImage(self):
        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        cv2.imwrite(filename, self.tmp)
        print('Image saved as:', self.filename)

    def set_image(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def runInference(self):
        cfg = Config.fromfile('mmdetection/configs/solov2/solov2_light_r18_fpn_3x_coco.py')
        cfg.model.mask_head.num_classes = 1
        checkpoint = 'hhn_solov2.pth'
        model = build_detector(cfg.model)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.cfg = cfg
        model.to('cuda')
        model.eval()
        self.model = model
        self.time_start = time.time()

        result = inference_detector(self.model, self.image)
        displayLabel = self.model.show_result(
            self.image,
            result,
            score_thr=self.thresholdValue,
            show=False,
            wait_time=0,
            win_name='result',
            bbox_color=None,
            text_color=(200, 200, 200),
            mask_color=None,
            out_file=None)
        self.time_detect = time.time() - self.time_start
        self.infTimeLabel.setText(str(self.time_detect))
        self.set_image(displayLabel)

        center_list = detect_center(self.image, result, self.thresholdValue)
        print(center_list)
        # process = Process(target=clientUtilities, args=[center_list])
        # process.start()

        for center in center_list:
            displayLabel = cv2.circle(displayLabel, center, 10, (0, 0, 255), -1)
        self.set_image(displayLabel)

    def runInferenceVideo(self):
        cfg = Config.fromfile('mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
        cfg.model.roi_head.bbox_head.num_classes = 1

        checkpoint = 'epoch_30.pth'
        model = build_detector(cfg.model)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.cfg = cfg
        model.to('cuda')
        model.eval()

        time_process_start = 0
        time_process_end = 0

        cap = cv2.VideoCapture(0)
        cap.set(3, 1080)
        cap.set(4, 640)
        while True:
            time_process_start = time.time()
            score_thr = 0.9
            _, img = cap.read()
            result = inference_detector(model, img)
            img_show = model.show_result(
                img,
                result,
                score_thr=score_thr,
                show=False,
                wait_time=0,
                win_name='result',
                bbox_color=None,
                text_color=(200, 200, 200),
                mask_color=None,
                out_file=None)
            center_list = detect_center_bbox(result, 0.9)
            for center in center_list:
                img_show = cv2.circle(img_show, center, 3, (0, 0, 255), -1)
            time_process_end = time.time() - time_process_start
            cv2.putText(img_show, 'time_process(s):' + str(np.round(time_process_end, 3)), (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 233), 2)
            self.set_image(img_show)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Start"))
        self.saveButton.setText(_translate("MainWindow", "Save"))
        self.runButton.setText(_translate("MainWindow", "Run"))
        self.label.setText(_translate("MainWindow", "Inference Time (s):"))
        self.infTimeLabel.setText(_translate("MainWindow", "Time"))
        self.displayLabel.setText(_translate("MainWindow", "Load Image to start"))
        self.label_2.setText(_translate("MainWindow", "0"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
