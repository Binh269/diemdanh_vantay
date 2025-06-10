# Hệ Thống Điểm Danh Tự Động

Hệ thống điểm danh tự động sử dụng công nghệ nhận diện khuôn mặt và vân tay, được phát triển để quản lý điểm danh trong môi trường giáo dục.

## Tính Năng Chính

### 1. Điểm Danh Tự Động
- Nhận diện khuôn mặt tự động
- Xác thực vân tay
- Điểm danh thủ công (dành cho giáo viên)

### 2. Quản Lý Lớp Học
- Tạo và quản lý lớp học phần
- Thêm/xóa sinh viên vào lớp
- Import/export danh sách sinh viên qua Excel
- Xem lịch học và lịch biểu

### 3. Quản Lý Người Dùng
- Phân quyền người dùng (Admin, Giáo viên, Sinh viên)
- Quản lý thông tin cá nhân
- Đăng ký và xác thực vân tay

### 4. Báo Cáo và Thống Kê
- Xem lịch sử điểm danh
- Thống kê chi tiết theo lớp, môn học
- Xuất báo cáo ra Excel

## Yêu Cầu Hệ Thống

### Phần Cứng
- Camera để nhận diện khuôn mặt
- Cảm biến vân tay Futronic (tùy chọn)

### Phần Mềm
- Python 3.x
- Django Framework
- Các thư viện xử lý ảnh và nhận diện khuôn mặt
- SDK Futronic cho xử lý vân tay

## Cài Đặt

1. Clone repository
2. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```
3. Cấu hình database
4. Chạy migrations:
```bash
python manage.py migrate
```
5. Khởi động server:
```bash
python manage.py runserver
```

## Hướng Dẫn Sử Dụng

### Cho Giáo Viên
1. Đăng nhập vào hệ thống
2. Tạo lớp học phần mới
3. Thêm sinh viên vào lớp
4. Thực hiện điểm danh tự động hoặc thủ công

### Cho Sinh Viên
1. Đăng nhập vào hệ thống
2. Đăng ký vân tay (nếu sử dụng)
3. Xem lịch học và lịch sử điểm danh

## Bảo Mật
- Xác thực người dùng qua login
- Phân quyền chi tiết
- Mã hóa dữ liệu nhạy cảm
- Bảo vệ API endpoints

## Hỗ Trợ
Nếu bạn gặp vấn đề hoặc cần hỗ trợ, vui lòng liên hệ với đội ngũ phát triển. 