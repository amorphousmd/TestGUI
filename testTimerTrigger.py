import os
import cv2

from apscheduler.schedulers.background import BackgroundScheduler

read_allow = False
timer_interval = 0.3

def print_out():
    global read_allow
    read_allow = True

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(print_out, 'interval', seconds=timer_interval)
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        if read_allow:
            ret, frame = cam.read()
            read_allow = False
        else:
            continue
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)

    cam.release()

    cv2.destroyAllWindows()