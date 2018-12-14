import cv2

#image_x, image_y = 100, 100

cap = cv2.VideoCapture(0)


def main():
    total_pics = 1000
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (image_x, image_y))
        cv2.imwrite("hand_images/" + str(pic_no) + ".jpg", frame)
        cv2.imshow("Capturing gesture", frame)
        pic_no += 1
        if pic_no == total_pics:
            break


main()
