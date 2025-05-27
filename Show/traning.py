# traning.py

import os
import sqlite3
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import json
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

def embedding_to_string(embedding):
    return json.dumps(embedding.tolist())

def connect_to_db():
    return sqlite3.connect("db.sqlite3")

def train_new_student(mssv, ten, lop, images):
    """
    Hàm đào tạo để thêm thông tin sinh viên và embedding khuôn mặt vào SQLite.
    Args:
        mssv (str): Mã số sinh viên
        ten (str): Tên sinh viên
        lop (str): Lớp của sinh viên
        images (list): Danh sách các file ảnh từ request.FILES
    Returns:
        tuple: (success: bool, message: str)
    """
    if len(images) < 5:
        return False, "Cần ít nhất 5 ảnh của sinh viên để training"

    embeddings_list = []
    for image in tqdm(images, desc=f"Processing {mssv}"):
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height  

                if area < 15000:
                    continue
                face_crop = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                face_tensor = transform(face_crop).unsqueeze(0).to(device)
                embedding = model(face_tensor).detach().cpu().numpy().flatten()
                embeddings_list.append(embedding)

    if not embeddings_list:
        return False, "Không tìm thấy khuôn mặt nào trong ảnh"

    avg_embedding = np.mean(embeddings_list, axis=0)
    embedding_str = embedding_to_string(avg_embedding)

    conn = connect_to_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO thongtin (mssv, name, lop, khuonmat)
            VALUES (?, ?, ?, ?)
        """, (mssv, ten, lop, embedding_str))
        conn.commit()
    except Exception as e:
        conn.close()
        return False, f"Lỗi khi lưu vào database: {str(e)}"
    finally:
        conn.close()

    return True, f"Thêm sinh viên {mssv} và embedding vào database thành công!"