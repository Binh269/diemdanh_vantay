import torch
from facenet_pytorch import InceptionResnetV1
import tensorflow as tf
import tensorflowjs as tfjs
import onnx
from onnx_tf.backend import prepare

# 1. Tải mô hình PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model.load_state_dict(torch.load('Show/models/face_recognition_model1.pth', map_location=device))

# 2. Xuất mô hình sang định dạng ONNX
dummy_input = torch.randn(1, 3, 160, 160).to(device)  # Input giả với kích thước 160x160
onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Model exported to ONNX format.")

# 3. Chuyển từ ONNX sang TensorFlow SavedModel
onnx_model = onnx.load(onnx_path)
tf_model = prepare(onnx_model)
tf_saved_model_path = "tf_model"
tf_model.export_graph(tf_saved_model_path)
print("Model converted to TensorFlow SavedModel.")

# 4. Chuyển từ SavedModel sang TensorFlow.js
tfjs_target_dir = "static/models/inception_resnet_v1"
tfjs.converters.save_keras_model(tf.keras.models.load_model(tf_saved_model_path), tfjs_target_dir)
print(f"Model converted to TensorFlow.js format and saved to {tfjs_target_dir}")
