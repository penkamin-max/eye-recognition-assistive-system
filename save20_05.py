import cv2
import mediapipe as mp
import numpy as np
import time
import webbrowser
import urllib.request 
from collections import deque 

# Khởi tạo thư viện MediaPipe Face Mesh gốc
mp_face_mesh = mp.solutions.face_mesh

# Định nghĩa các điểm mốc (Indices) của hốc mắt và con ngươi
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_CENTER = 468 
RIGHT_IRIS_CENTER = 473 

# --- CÁC THÔNG SỐ TỐI ƯU HỆ THỐNG ---
DWELL_THRESHOLD = 1.36  # Thời gian chờ
EAR_THRESHOLD = 0.20    # Ngưỡng nháy mắt
WINK_CONSEC_FRAMES = 2  # Khung hình kích hoạt nháy mắt
SMOOTHING_WINDOW_SIZE = 15 # Bộ giảm xóc

# --- THIẾT LẬP GIAO DIỆN CHUẨN 16:9 (HD 720p) ---
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# Tọa độ các khối UI
BOX_SIZE = 350
GAP = (CANVAS_WIDTH - (2 * BOX_SIZE)) // 3 
Y_TOP = (CANVAS_HEIGHT - BOX_SIZE) // 2 

ZONE_YOUTUBE = (GAP, Y_TOP, GAP + BOX_SIZE, Y_TOP + BOX_SIZE) 
ZONE_FACEBOOK = (GAP * 2 + BOX_SIZE, Y_TOP, GAP * 2 + 2 * BOX_SIZE, Y_TOP + BOX_SIZE)

# Tọa độ nút BẮT ĐẦU ở màn hình chờ (Căn giữa)
START_W, START_H = 400, 200
START_X1 = (CANVAS_WIDTH - START_W) // 2
START_Y1 = (CANVAS_HEIGHT - START_H) // 2
ZONE_START = (START_X1, START_Y1, START_X1 + START_W, START_Y1 + START_H)

# --- LINK LOGO ONLINE ---
YOUTUBE_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png"
FACEBOOK_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Logo_de_Facebook.png/1280px-Logo_de_Facebook.png"

class GazeSmoother:
    def __init__(self, size=SMOOTHING_WINDOW_SIZE):
        self.queue_x = deque(maxlen=size)
        self.queue_y = deque(maxlen=size)
        
    def get_smooth_coordinates(self, x, y):
        self.queue_x.append(x)
        self.queue_y.append(y)
        smooth_x = int(sum(self.queue_x) / len(self.queue_x))
        smooth_y = int(sum(self.queue_y) / len(self.queue_y))
        return smooth_x, smooth_y

def calculate_ear(eye_points, landmarks, width, height):
    coords = np.array([(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in eye_points])
    v1 = np.linalg.norm(coords[1] - coords[4])
    v2 = np.linalg.norm(coords[2] - coords[5])
    h = np.linalg.norm(coords[0] - coords[3])
    if h == 0: return 0.5
    return (v1 + v2) / (2.0 * h)

def calculate_gaze_ratios(landmarks, width, height):
    left_coords = np.array([(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in LEFT_EYE_LANDMARKS])
    left_iris_x = int(landmarks[LEFT_IRIS_CENTER].x * width)
    left_iris_y = int(landmarks[LEFT_IRIS_CENTER].y * height)
    l_min_x, l_max_x = np.min(left_coords[:, 0]), np.max(left_coords[:, 0])
    l_min_y, l_max_y = np.min(left_coords[:, 1]), np.max(left_coords[:, 1])
    
    right_coords = np.array([(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in RIGHT_EYE_LANDMARKS])
    right_iris_x = int(landmarks[RIGHT_IRIS_CENTER].x * width)
    right_iris_y = int(landmarks[RIGHT_IRIS_CENTER].y * height)
    r_min_x, r_max_x = np.min(right_coords[:, 0]), np.max(right_coords[:, 0])
    r_min_y, r_max_y = np.min(right_coords[:, 1]), np.max(right_coords[:, 1])
    
    if (l_max_x - l_min_x) == 0 or (r_max_x - r_min_x) == 0: return 0.5, 0.5
    if (l_max_y - l_min_y) == 0 or (r_max_y - r_min_y) == 0: return 0.5, 0.5
    
    l_ratio_x = (left_iris_x - l_min_x) / (l_max_x - l_min_x)
    r_ratio_x = (right_iris_x - r_min_x) / (r_max_x - r_min_x)
    ratio_x = (l_ratio_x + r_ratio_x) / 2.0
    
    l_ratio_y = (left_iris_y - l_min_y) / (l_max_y - l_min_y)
    r_ratio_y = (right_iris_y - r_min_y) / (r_max_y - r_min_y)
    ratio_y = (l_ratio_y + r_ratio_y) / 2.0
    
    return ratio_x, ratio_y

def load_logo_from_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except: return None

def overlay_image_alpha(img, img_overlay, x, y):
    if img_overlay is None: return img
    if img_overlay.shape[2] == 3:
        h_o, w_o = img_overlay.shape[:2]
        img[y:y+h_o, x:x+w_o] = img_overlay
        return img
        
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: return img

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = img_overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] + alpha_inv * img_crop[:, :, c])
    return img

def main():
    print("Đang tải logo từ Internet...")
    logo_yt_raw = load_logo_from_url(YOUTUBE_LOGO_URL)
    logo_fb_raw = load_logo_from_url(FACEBOOK_LOGO_URL)
    
    logo_yt_resized = cv2.resize(logo_yt_raw, (200, 200)) if logo_yt_raw is not None else None
    logo_fb_resized = cv2.resize(logo_fb_raw, (200, 200)) if logo_fb_raw is not None else None
    
    cap = cv2.VideoCapture(0)
    gaze_smoother = GazeSmoother() 
    
    window_name = "Dashboard Anh Mat - Optimised by BME HUST"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # --- BIẾN ĐIỀU KHIỂN LUỒNG (STATE MACHINE) ---
    app_state = "START_SCREEN" # Trạng thái ban đầu
    current_target = None
    start_look_time = None
    
    system_paused = False
    left_wink_counter = 0
    right_wink_counter = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) as face_mesh:
        
        print("Hệ thống đã sẵn sàng!")
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
                
            frame = cv2.flip(frame, 1) 
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            gaze_x, gaze_y = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2
            detected_zone = None
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks, w, h)
                right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks, w, h)
                
                # Logic Nháy mắt
                if left_ear < EAR_THRESHOLD and right_ear > (EAR_THRESHOLD + 0.05):
                    left_wink_counter += 1
                else:
                    if left_wink_counter >= WINK_CONSEC_FRAMES:
                        system_paused = not system_paused
                    left_wink_counter = 0

                if right_ear < EAR_THRESHOLD and left_ear > (EAR_THRESHOLD + 0.05):
                    right_wink_counter += 1
                else:
                    if right_wink_counter >= WINK_CONSEC_FRAMES: break
                    right_wink_counter = 0
                
                if not system_paused:
                    ratio_x, ratio_y = calculate_gaze_ratios(landmarks, w, h)
                    raw_x = int(np.interp(ratio_x, [0.35, 0.65], [100, CANVAS_WIDTH - 100]))
                    raw_y = int(np.interp(ratio_y, [0.30, 0.70], [100, CANVAS_HEIGHT - 100]))
                    gaze_x, gaze_y = gaze_smoother.get_smooth_coordinates(raw_x, raw_y)
                    
                    # Cập nhật vùng nhìn dựa trên Trạng thái màn hình
                    if app_state == "START_SCREEN":
                        if ZONE_START[0] <= gaze_x <= ZONE_START[2] and ZONE_START[1] <= gaze_y <= ZONE_START[3]:
                            detected_zone = "START"
                            
                    elif app_state == "MAIN_MENU":
                        if ZONE_YOUTUBE[0] <= gaze_x <= ZONE_YOUTUBE[2] and ZONE_YOUTUBE[1] <= gaze_y <= ZONE_YOUTUBE[3]:
                            detected_zone = "YOUTUBE"
                        elif ZONE_FACEBOOK[0] <= gaze_x <= ZONE_FACEBOOK[2] and ZONE_FACEBOOK[1] <= gaze_y <= ZONE_FACEBOOK[3]:
                            detected_zone = "FACEBOOK"

            # --- VẼ GIAO DIỆN THEO TRẠNG THÁI ---
            canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255
            
            # Căn giữa Tiêu đề chung
            title_text = "HE THONG DIEU KHIEN ANH MAT - BME HUST"
            text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            title_x = (CANVAS_WIDTH - text_size[0]) // 2
            cv2.putText(canvas, title_text, (title_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

            # ----------------------------------------------------
            # MÀN HÌNH CHỜ (START SCREEN)
            # ----------------------------------------------------
            if app_state == "START_SCREEN":
                color_start = (0, 150, 0) # Xanh lá mặc định
                
                if detected_zone == "START":
                    color_start = (0, 200, 0) # Sáng lên khi nhìn vào
                    cv2.rectangle(canvas, (ZONE_START[0]-10, ZONE_START[1]-10), (ZONE_START[2]+10, ZONE_START[3]+10), (0, 215, 255), 6)
                
                cv2.rectangle(canvas, (ZONE_START[0], ZONE_START[1]), (ZONE_START[2], ZONE_START[3]), color_start, -1)
                
                start_txt = "BAT DAU"
                start_size = cv2.getTextSize(start_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
                cv2.putText(canvas, start_txt, (ZONE_START[0] + (START_W - start_size[0])//2, ZONE_START[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            
            # ----------------------------------------------------
            # MÀN HÌNH CHỌN APP (MAIN MENU)
            # ----------------------------------------------------
            elif app_state == "MAIN_MENU":
                color_yt = (0, 0, 180) 
                color_fb = (180, 70, 0) 
                
                if detected_zone == "YOUTUBE":
                    color_yt = (0, 0, 255) 
                    cv2.rectangle(canvas, (ZONE_YOUTUBE[0]-10, ZONE_YOUTUBE[1]-10), (ZONE_YOUTUBE[2]+10, ZONE_YOUTUBE[3]+10), (0, 215, 255), 6) 
                elif detected_zone == "FACEBOOK":
                    color_fb = (255, 120, 0) 
                    cv2.rectangle(canvas, (ZONE_FACEBOOK[0]-10, ZONE_FACEBOOK[1]-10), (ZONE_FACEBOOK[2]+10, ZONE_FACEBOOK[3]+10), (0, 215, 255), 6) 

                cv2.rectangle(canvas, (ZONE_YOUTUBE[0], ZONE_YOUTUBE[1]), (ZONE_YOUTUBE[2], ZONE_YOUTUBE[3]), color_yt, -1)
                if logo_yt_resized is not None:
                    canvas = overlay_image_alpha(canvas, logo_yt_resized, ZONE_YOUTUBE[0] + (BOX_SIZE - 200)//2, ZONE_YOUTUBE[1] + (BOX_SIZE - 200)//2)
                
                cv2.rectangle(canvas, (ZONE_FACEBOOK[0], ZONE_FACEBOOK[1]), (ZONE_FACEBOOK[2], ZONE_FACEBOOK[3]), color_fb, -1)
                if logo_fb_resized is not None:
                    canvas = overlay_image_alpha(canvas, logo_fb_resized, ZONE_FACEBOOK[0] + (BOX_SIZE - 200)//2, ZONE_FACEBOOK[1] + (BOX_SIZE - 200)//2)

            # --- LOGIC ĐIỀU KHIỂN & ĐẾM GIỜ ---
            if not system_paused and results.multi_face_landmarks:
                cv2.circle(canvas, (gaze_x, gaze_y), 15, (255, 0, 0), -1) # Vẽ con trỏ
                
                if detected_zone:
                    if current_target != detected_zone:
                        current_target = detected_zone
                        start_look_time = time.time()
                    else:
                        elapsed = time.time() - start_look_time
                        
                        # Vẽ chữ đếm ngược
                        if app_state == "START_SCREEN":
                            dwell_text = f"Mo khoa: {DWELL_THRESHOLD - elapsed:.1f}s"
                        else:
                            dwell_text = f"Dang mo: {DWELL_THRESHOLD - elapsed:.1f}s"
                            
                        dwell_size = cv2.getTextSize(dwell_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        cv2.putText(canvas, dwell_text, ((CANVAS_WIDTH - dwell_size[0]) // 2, CANVAS_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        
                        # Kích hoạt hành động
                        if elapsed >= DWELL_THRESHOLD:
                            if app_state == "START_SCREEN":
                                app_state = "MAIN_MENU" # Vào màn hình chính
                                current_target = None
                                
                            elif app_state == "MAIN_MENU":
                                if current_target == "YOUTUBE": webbrowser.open("https://www.youtube.com/")
                                elif current_target == "FACEBOOK": webbrowser.open("https://www.facebook.com/")
                                
                                # MỞ APP XONG LẬP TỨC TRỞ VỀ MÀN HÌNH CHỜ (Chống Midas Touch)
                                app_state = "START_SCREEN"
                                current_target = None
                else:
                    current_target = None
                    start_look_time = None
            
            # Tích hợp Webcam
            webcam_w, webcam_h = 240, 180
            webcam_resized = cv2.resize(frame, (webcam_w, webcam_h))
            canvas[CANVAS_HEIGHT - webcam_h - 15 : CANVAS_HEIGHT - 15, CANVAS_WIDTH - webcam_w - 15 : CANVAS_WIDTH - 15] = webcam_resized
            cv2.rectangle(canvas, (CANVAS_WIDTH - webcam_w - 20, CANVAS_HEIGHT - webcam_h - 20), (CANVAS_WIDTH - 10, CANVAS_HEIGHT - 10), (0, 255, 0), 3)
            
            # Màn hình Tạm dừng
            if system_paused:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
                pause_text = "TAM DUNG - NHAY MAT TRAI DE TIEP TUC"
                pause_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.putText(canvas, pause_text, ((CANVAS_WIDTH - pause_size[0]) // 2, CANVAS_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()