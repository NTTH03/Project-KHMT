# face_train.py
import cv2
import numpy as np
import face_recognition
import os
import pickle

DUONG_DAN_THU_MUC = "pic"
HINH_ANH = []
TEN_LOP = []

try:
    danh_sach_lop = os.listdir(DUONG_DAN_THU_MUC)
    print(f"Tìm thấy các lớp: {danh_sach_lop}")
except Exception as e:
    raise ValueError(f"Lỗi khi truy cập thư mục {DUONG_DAN_THU_MUC}: {str(e)}")

for ten_lop in danh_sach_lop:
    duong_dan_lop = os.path.join(DUONG_DAN_THU_MUC, ten_lop)
    if not os.path.isdir(duong_dan_lop):
        continue  # Bỏ qua nếu không phải thư mục

    danh_sach_anh = os.listdir(duong_dan_lop)
    for ten_anh in danh_sach_anh:
        duong_dan_anh = os.path.join(duong_dan_lop, ten_anh)
        anh = cv2.imread(duong_dan_anh)
        if anh is not None:
            HINH_ANH.append(anh)
            TEN_LOP.append(ten_lop)
        else:
            print(f"Lỗi khi đọc ảnh: {duong_dan_anh}")

print(f"Đã đọc tổng cộng {len(HINH_ANH)} ảnh.")

def ma_hoa_khuon_mat(hinh_anh):
    danh_sach_ma_hoa = []
    ten_lop_da_ma_hoa = []
    for i, anh in enumerate(hinh_anh):
        anh = cv2.resize(anh, (400, 400))  # Bình thường hóa kích thước
        rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
        ma_hoa = face_recognition.face_encodings(rgb)
        if ma_hoa:
            danh_sach_ma_hoa.append(ma_hoa[0])
            ten_lop_da_ma_hoa.append(TEN_LOP[i])
        else:
            print(f"Ảnh {ten_anh} không tìm thấy khuôn mặt, bỏ qua.")
    return danh_sach_ma_hoa, ten_lop_da_ma_hoa

DANH_SACH_MA_HOA, TEN_LOP_DA_MA_HOA = ma_hoa_khuon_mat(HINH_ANH)
print(f"Đã mã hóa {len(DANH_SACH_MA_HOA)} khuôn mặt hợp lệ.")

# Lưu kết quả
with open("mahoa.pkl", "wb") as f:
    pickle.dump((DANH_SACH_MA_HOA, TEN_LOP_DA_MA_HOA), f)
print("Đã lưu dữ liệu mã hóa vào mahoa.pkl")
