import cv2
import numpy as np
import face_recognition
import os
import pickle

# Thư mục chứa ảnh
DUONG_DAN_THU_MUC = "pic"
HINH_ANH = []
TEN_LOP = []

# Kiểm tra ảnh bị mờ
def is_blurry(image, threshold=50.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Đọc danh sách lớp (thư mục con)
try:
    danh_sach_lop = os.listdir(DUONG_DAN_THU_MUC)
    print(f"Tìm thấy các lớp: {danh_sach_lop}")
except Exception as e:
    raise ValueError(f"Lỗi khi truy cập thư mục {DUONG_DAN_THU_MUC}: {str(e)}")

# Đọc và lọc ảnh từ từng lớp
for ten_lop in danh_sach_lop:
    duong_dan_lop = os.path.join(DUONG_DAN_THU_MUC, ten_lop)
    if not os.path.isdir(duong_dan_lop):
        continue  # Bỏ qua nếu không phải thư mục

    danh_sach_anh = os.listdir(duong_dan_lop)
    for ten_anh in danh_sach_anh:
        duong_dan_anh = os.path.join(duong_dan_lop, ten_anh)
        anh = cv2.imread(duong_dan_anh)

        if anh is None or anh.shape[0] < 200 or anh.shape[1] < 200:
            print(f"Ảnh {duong_dan_anh} không hợp lệ , bỏ qua.")
            continue

        if is_blurry(anh):
            print(f"Ảnh {duong_dan_anh} bị mờ, bỏ qua.")
            continue

        HINH_ANH.append(anh)
        TEN_LOP.append(ten_lop)

print(f"Đã đọc tổng cộng {len(HINH_ANH)} ảnh hợp lệ sau khi lọc.")

# Mã hóa khuôn mặt
def ma_hoa_khuon_mat(hinh_anh):
    danh_sach_ma_hoa = []
    ten_lop_da_ma_hoa = []
    for i, img in enumerate(hinh_anh):
        img = cv2.resize(img, (400, 400))  # Chuẩn hóa kích thước
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ma_hoa = face_recognition.face_encodings(rgb)

        if ma_hoa:
            danh_sach_ma_hoa.append(ma_hoa[0])
            ten_lop_da_ma_hoa.append(TEN_LOP[i])
        else:
            print(f"Ảnh {i} không tìm thấy khuôn mặt, bỏ qua.")

    return danh_sach_ma_hoa, ten_lop_da_ma_hoa

# Danh sách mã hóa khuôn mặt
DANH_SACH_MA_HOA, TEN_LOP_DA_MA_HOA = ma_hoa_khuon_mat(HINH_ANH)
print(f"Đã mã hóa thành công {len(DANH_SACH_MA_HOA)} khuôn mặt.")

# Lưu dữ liệu mã hóa
with open("mahoa.pkl", "wb") as f:
    pickle.dump((DANH_SACH_MA_HOA, TEN_LOP_DA_MA_HOA), f)
print("Đã lưu dữ liệu mã hóa vào 'mahoa.pkl'")
