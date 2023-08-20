import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from sklearn.cluster import KMeans

def set_up_pose_detection_model():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    return mp_pose, mp_drawing

def get_video_writer(image_name, video_path):
    basename = os.path.basename(video_path)
    filename, extension = os.path.splitext(basename)
    size = (480, 640)
    make_directory(image_name)
    out = cv2.VideoWriter(f"{image_name}/{filename}_out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, size)
    return out

def make_directory(name: str):
    if not os.path.isdir(name):
        os.mkdir(name)


def resize_image(image):
    h, w, _ = image.shape
    h, w = h // 2, w // 2
    image = cv2.resize(image, (w, h))
    return image, h, w

def pose_process_image(image, pose):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def plot_angles_from_frames(mp_pose, landmarks, image, h, w, max_angle_right=0, ):
    angles = []
    val = 50
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_WRIST.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value,
                              mp_pose.PoseLandmark.LEFT_ANKLE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value,
                              mp_pose.PoseLandmark.RIGHT_ANKLE.value, landmarks, image, h, w - val)
    angles.append(angle)

    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value, landmarks, image, h, w - val)
    angles.append(angle)

    angle, imge = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                             mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                             mp_pose.PoseLandmark.LEFT_HIP.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                       mp_pose.PoseLandmark.RIGHT_HIP.value, landmarks, image, h,
                                                       w - val)

    angles.append(angle_wrist_shoulder_hip_right)
    max_angle_right = max(max_angle_right, angle_wrist_shoulder_hip_right)

    return angles, max_angle_right


def plot_angle(p1, p2, p3, landmarks, image, h, w):
    a = [landmarks[p1].x,
         landmarks[p1].y]
    b = [landmarks[p2].x, landmarks[p2].y]
    c = [landmarks[p3].x, landmarks[p3].y]
    angle = calculate_angle(a, b, c)
    draw_angle(tuple(np.multiply(b, [w, h]).astype(int)), image, round(angle))
    return angle, image


def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)
    c = np.array(c) 
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)


def draw_angle(org: tuple, image, angle):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255, 255, 255)
    thickness = 1
    image = cv2.putText(image, str(angle), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def add_stage(frames, max_value):
    stage = 1
    for frame in frames:
        if frame[-1] == max_value:
            stage = 0
        frame.append(stage)
    return frames


def draw_landmarks(results, mp_drawing, mp_pose, image):
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:
            results.pose_landmarks.landmark[idx].visibility = 0
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    return image

def get_frames_angles(image_name: str, video_path: str) -> tuple:
    mp_pose, mp_drawing = set_up_pose_detection_model()
    cap = cv2.VideoCapture(video_path)
    out = get_video_writer(image_name, video_path)
    img_count = 0
    output_images = []
    frames = []
    max_angle_right = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            image, h, w = resize_image(image)

            image, results = pose_process_image(image, pose)
            try:
                landmarks = results.pose_landmarks.landmark
                angles, max_angle_right = plot_angles_from_frames(mp_pose, landmarks, image, h, w, max_angle_right)
                frames.append(angles)

                image = draw_landmarks(results, mp_drawing, mp_pose, image)
                out.write(image)

                cv2.imshow('Image' , image)
                outImageFile = f"{image_name}/{image_name}{img_count}.jpg"
                cv2.imwrite(outImageFile, image)
                img_count += 1

            except:
                pass

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    out.release()

    return frames, max_angle_right


def kmeans(link1, link2,change):
    coach_frames, max_angle_right = get_frames_angles(image_name= 'coach', video_path= link1)
    coach_frames = add_stage(coach_frames, max_angle_right)
    student_frames,max_angle_right = get_frames_angles(image_name= 'student', video_path= link2)
    student_frames = add_stage(student_frames, max_angle_right)
    student_n_cluster = 15  # 0, 1, 2, 3
    X = np.array(student_frames)
    kmeans_student = KMeans(n_clusters=student_n_cluster, random_state=0).fit(X)
    def get_nearest_neighbor(image,indexes, frames):
        a = np.array(image)
        min_dist = sys.maxsize
        nearest = indexes[0]
        for idx in indexes:
            b = np.array(frames[idx])
            distance = np.linalg.norm(a-b)
            if(distance < min_dist):
                nearest = b
                min_dist = distance
        return nearest
    n_cluster_coach = 15
    X = np.array(coach_frames)
    kmeans_coach = KMeans(n_clusters = n_cluster_coach, random_state = 0).fit(X)
    student_cluster = []
    start = 0
    for i in range(1,len(kmeans_student.labels_)):
        if (kmeans_student.labels_[i] != kmeans_student.labels_[i - 1]):
            student_cluster.append({'label': kmeans_student.labels_[i - 1], 'start': start, 'end': i - 1})
            start = i
    vals=[]
    for label in (student_cluster):
        index_student = (label['start'] + label['end']) // 2
        student_image = student_frames[index_student]
        predict = kmeans_coach.predict([student_image]) 
        indexes_frame = np.where(kmeans_coach.labels_ == predict[0])
        nearest = get_nearest_neighbor(student_image, indexes_frame[0],coach_frames)
        new=[]
        new.append(list(nearest))
        new.append(list(student_image))
        vals.append(new)
        change.append(list(np.subtract(nearest,student_image)))
    return change,vals
