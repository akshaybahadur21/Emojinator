import cv2
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

image_x, image_y = 200, 200

cap = cv2.VideoCapture(0)


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(g_id):
    total_pics = 1200
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
    cap = cv2.VideoCapture(0)
    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image_orig = cv2.flip(image, 1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
        res = cv2.bitwise_and(image, cv2.bitwise_not(image_orig))

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # if cv2.contourArea(contour) > 10000 and frames > 50:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            pic_no += 1
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            save_img = gray[y1:y1 + h1, x1:x1 + w1]
            save_img = cv2.resize(save_img, (image_x, image_y))
            cv2.putText(image, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
            cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                if flag_start_capturing == False:
                    flag_start_capturing = True
                else:
                    flag_start_capturing = False
                    frames = 0
            if flag_start_capturing == True:
                frames += 1
            if pic_no == total_pics:
                break
            cv2.imshow("Capturing gesture", image)
            cv2.imshow("Res", res)


g_id = input("Enter gesture number: ")
main(g_id)
