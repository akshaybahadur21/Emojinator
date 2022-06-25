import cv2
import os
import mediapipe as mp
from keras.models import load_model
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model = load_model('emojinator_v3.h5')
emo_dict = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9}


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def get_emojis():
    emojis_folder = '/Users/3482704/Akshay_git/Emojinator/Emojinator_V3/hand_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder + str(emoji) + '.png', -1))
    return emojis


emojis = get_emojis()
image_x, image_y = 200, 200

cap = cv2.VideoCapture(0)


def main():
    total_pics = 1200
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
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
            contours = sorted(contours, key=cv2.contourArea)
            contour = contours[-1]
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            save_img = gray[y1:y1 + h1, x1:x1 + w1]
            save_img = cv2.resize(save_img, (image_x, image_y))
            pred_probab, pred_class = keras_predict(model, save_img)
            print(pred_class, pred_probab)
            image = overlay(image, emojis[emo_dict[pred_class]], x1 + 70, y1 - 120, 90, 90)
            if len(contours) > 1 and cv2.contourArea(contours[-2]) > 500:
                contour2 = contours[-2]
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                save_img2 = gray[y2:y2 + h2, x2:x2 + w2]
                save_img2 = cv2.resize(save_img2, (image_x, image_y))
                pred_probab, pred_class = keras_predict(model, save_img2)
                print(pred_class, pred_probab)
                image = overlay(image, emojis[emo_dict[pred_class]], x2 + 70, y2 - 120, 90, 90)
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
            image = rescale_frame(image, percent=75)
            cv2.imshow("Img", image)

    hands.close()
    cap.release()


def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y + h, x:x + w] = blend_transparent(image[y:y + h, x:x + w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


main()
