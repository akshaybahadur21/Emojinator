import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
emojis = [cv2.imread("/Users/3482704/Akshay_git/Emojinator/Emojinator_V3/other_emo/goggles.png", -1)]


def print_hi():
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        success, image = cap.read()
        image_orig = cv2.flip(image, 1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=face_landmark_drawing_spec,
                    connection_drawing_spec=face_connection_drawing_spec)
        res = cv2.bitwise_and(image, cv2.bitwise_not(image_orig))
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contours.sort(key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(contours[-1])
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            image_orig = overlay(image_orig, emojis[0], x1 - 10, y1 - 50, w1 + 20, h1 - 10)
        cv2.imshow('MediaPipe FaceMesh', image_orig)
        # cv2.imshow('MediaPipe FaceMesh', image_orig)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    face_mesh.close()
    cap.release()


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
