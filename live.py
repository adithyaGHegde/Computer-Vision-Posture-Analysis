import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from keras.models import load_model
import simplejpeg
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle,2)
def getCurrentPred(last_five_frames_df):
    good_video_count = 0
    y_pred_frames = loaded_model.predict(last_five_frames_df)
    for y_pred in y_pred_frames:
        if(y_pred > 0.5):
            good_video_count += 1
    if(good_video_count>3):
        return 1
    else:
        return 0
def captureVideoDataPred():
    mpDraw = mp.solutions.drawing_utils
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()
    cap = cv2.VideoCapture(0)
    df = pd.DataFrame({
        "angle_elbow_right" : [],
        "angle_knee_right" : [],
        "angle_shoulder_right": [],
        "angle_hip_right": [],
        "angle_elbow_left" : [],
        "angle_knee_left" : [],
        "angle_shoulder_left": [],
        "angle_hip_left": []
    })
    i = 0
    while True:
        frame_exists, frame = cap.read()
        if frame_exists == True:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                i = i+1
                landmarks = results.pose_landmarks.landmark
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                angle_elbow_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                angle_elbow_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                angle_knee_left = calculate_angle(hip_left, knee_left, ankle_left)
                angle_knee_right = calculate_angle(hip_right, knee_right, ankle_right)
                angle_shoulder_right = calculate_angle(hip_right,shoulder_right,elbow_right)
                angle_shoulder_left = calculate_angle(hip_left,shoulder_left,elbow_left)
                angle_hip_right = calculate_angle(knee_right,hip_right,shoulder_right)
                angle_hip_left = calculate_angle(knee_left,hip_left,shoulder_left)
                df.loc[i] = [angle_elbow_right, angle_knee_right, angle_shoulder_right,angle_hip_right,angle_elbow_left,angle_knee_left,angle_shoulder_left,angle_hip_left]
                if(i>=5):
                    current_frame_pred = getCurrentPred(df.loc[i-5:i])
                    if(current_frame_pred==0):
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2))
                        cv2.putText(frame, "Bad posture",
                            tuple(np.multiply(elbow_right[:2], [width,height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA )
                        new_frame = cv2.imencode('.jpg', frame)[1].tobytes()
                        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
                        key = cv2.waitKey(20)
                        if key == 27:
                            break
                    else:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2)
                        )
                        cv2.putText(frame, "Good posture",
                            tuple(np.multiply(elbow_right[:2], [width,height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA )
                        new_frame =  simplejpeg.encode_jpeg()('.jpg', frame)[1].tobytes()
                        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
                        key = cv2.waitKey(20)
                        if key == 27:
                            break
            except:
                print("No pose variable: Could be due to unfamiliar camera angle/unclear recording")
                pass              
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2)
                )
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break        
mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 
loaded_model = load_model('ML_model.h5')