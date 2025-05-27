import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TripletMarginLoss
import torch.optim as optim

# Định nghĩa mô hình CNN sâu hơn
class DeepFaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(DeepFaceRecognitionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 10 * 10, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        identity = F.interpolate(self.conv3(x), size=x.shape[2:], mode='nearest')
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        identity = F.interpolate(self.conv5(x), size=x.shape[2:], mode='nearest')
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        identity = F.interpolate(self.conv7(x), size=x.shape[2:], mode='nearest')
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.bn8(self.conv8(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 512 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Dataset tùy chỉnh
class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for idx, label in enumerate(os.listdir(root_dir)):
            person_dir = os.path.join(root_dir, label)
            if os.path.isdir(person_dir):
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        if image is not None:
            face_crop, _ = detect_and_crop_face(image)
            if face_crop is not None:
                face_tensor = preprocess_image(face_crop)
                return face_tensor, label
        return None, None

# Hàm tiền xử lý
def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_crop = image[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crop = face_crop.astype(np.float32) / 127.5 - 1
        return face_crop, (x, y, x+w, y+h)
    return None, None

def preprocess_image(image):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

def custom_collate_fn(batch):
    filtered_batch = [sample for sample in batch if sample[0] is not None]
    if not filtered_batch:
        return None
    images, labels = zip(*filtered_batch)
    return torch.stack(images, dim=0).squeeze(1), torch.tensor(labels)

def train_face_model(model, dataset_dir, model_path, epochs=30, batch_size=32):
    dataset = FaceDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)

    criterion = TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Giảm learning rate để học chi tiết hơn

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        triplets_processed = 0
        
        for batch in dataloader:
            if batch is None:
                continue
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            anchor_idx = []
            positive_idx = []
            negative_idx = []
            
            for i, label in enumerate(labels):
                positive_candidates = [j for j, l in enumerate(labels) if l == label and j != i]
                negative_candidates = [j for j, l in enumerate(labels) if l != label]
                if positive_candidates and negative_candidates:
                    anchor_idx.append(i)
                    positive_idx.append(np.random.choice(positive_candidates))
                    negative_idx.append(np.random.choice(negative_candidates))

            if not anchor_idx:
                continue

            anchor = images[anchor_idx]
            positive = images[positive_idx]
            negative = images[negative_idx]

            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            triplets_processed += 1

        if triplets_processed > 0:
            avg_loss = running_loss / triplets_processed
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: Không đủ dữ liệu để tạo triplet")

    torch.save(model.state_dict(), model_path)
    print(f"Đã lưu mô hình huấn luyện tại {model_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFaceRecognitionCNN().to(device)
    DATASET_DIR = "../dataset"
    MODEL_PATH = "Show/models/deep_face_recognition_model.pth"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Đã tải mô hình từ {MODEL_PATH}, tiếp tục huấn luyện...")
        except RuntimeError as e:
            print(f"Lỗi khi tải state_dict: {e}")
            print("Tạo mới state_dict cho DeepFaceRecognitionCNN...")

    train_face_model(model, DATASET_DIR, MODEL_PATH, epochs=30, batch_size=32)