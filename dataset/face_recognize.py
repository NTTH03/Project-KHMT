import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
import os

# Tải dữ liệu mã hóa 
with open("mahoa.pkl", "rb") as f:
    DANH_SACH_MA_HOA, TEN_LOP = pickle.load(f)

def ghi_danh_sach(name):
    ngay_hien_tai = datetime.now().strftime("%Y-%m-%d")
    ten_file = f"thamdu_{ngay_hien_tai}.csv"

    # Tạo file
    if not os.path.exists(ten_file):
        with open(ten_file, "w", encoding="utf-8") as f:
            f.write("Name,Time\n")  # Chỉ có Name và Time

    # Mở file ở chế độ append (thêm vào cuối)
    with open(ten_file, "a+", encoding="utf-8") as f:
        thoi_gian = datetime.now().strftime("%H:%M:%S")
        f.writelines(f"{name},{thoi_gian}\n")  # Chỉ ghi tên và giờ
        print(f"Ghi danh: {name} lúc {thoi_gian}")

# Mở camera
cap = cv2.VideoCapture("videocheck.mp4")
if not cap.isOpened():
    raise ValueError("Không thể mở camera.")

so_khung = 0
xu_ly_moi_n_khung_hinh = 8
thoi_gian_cho_phep_ghi_lai = 100

vi_tri_khuon_mat = []
ma_hoa_khung_hinh = []
thoi_gian_lan_cuoi_ghi_danh = {}  # Dictionary để theo dõi thời gian ghi danh cuối cùng của mỗi người

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình.")
            break

        # Resize để tăng tốc xử lý
        khung_nho = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        rgb_khung = cv2.cvtColor(khung_nho, cv2.COLOR_BGR2RGB)

        # Xử lý mỗi n khung hình
        if so_khung % xu_ly_moi_n_khung_hinh == 0:
            vi_tri_khuon_mat = face_recognition.face_locations(rgb_khung, model="hog")
            ma_hoa_khung_hinh = face_recognition.face_encodings(rgb_khung, vi_tri_khuon_mat)
        so_khung += 1

        # Nhận diện khuôn mặt
        for ma_hoa, vi_tri in zip(ma_hoa_khung_hinh, vi_tri_khuon_mat):
            ket_qua = face_recognition.compare_faces(DANH_SACH_MA_HOA, ma_hoa)
            khoang_cach = face_recognition.face_distance(DANH_SACH_MA_HOA, ma_hoa)

            name = "Unknown"
            if True in ket_qua:
                chi_so = np.argmin(khoang_cach)
                if khoang_cach[chi_so] < 0.5:
                    name = TEN_LOP[chi_so].upper()
                    thoi_gian_hien_tai = datetime.now()
                    # Kiểm tra xem đã ghi danh người này gần đây chưa
                    if name not in thoi_gian_lan_cuoi_ghi_danh or \
                            (thoi_gian_hien_tai - thoi_gian_lan_cuoi_ghi_danh[name]).total_seconds() >= thoi_gian_cho_phep_ghi_lai:
                        ghi_danh_sach(name)
                        thoi_gian_lan_cuoi_ghi_danh[name] = thoi_gian_hien_tai  # Cập nhật thời gian ghi danh cuối cùng

            # Vẽ khung mặt và tên
            y1, x2, y2, x1 = [v * 2.5 for v in vi_tri]  # 2.5 để bù cho resize 0.4
            y1, x2, y2, x1 = map(int, (y1, x2, y2, x1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Nhan dien khuon mat", frame)

        # Nhấn Q để đóng cửa sổ
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        break

cap.release()
cv2.destroyAllWindows()
print("Đã đóng camera.")
