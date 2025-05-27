from django.utils import timezone
from django.db import models
# Phòng học
class Model_Phong(models.Model):
    maPhong = models.CharField(max_length=20, unique=True)  # Mã phòng học duy nhất
    tenPhong = models.CharField(max_length=255)
    trangThaiPhong = models.CharField(max_length=255)

    def __str__(self):
        return self.maPhong

# Thiết bị
class Model_ThietBi(models.Model):
    maThietBi = models.CharField(max_length=20, unique=True)  # Mã thiết bị duy nhất
    tenThietBi = models.CharField(max_length=255)
    trangThaiThietBi = models.CharField(max_length=255)
    phongHoc = models.ForeignKey(Model_Phong, related_name='thiet_bis', on_delete=models.CASCADE)

    def __str__(self):
        return self.maThietBi
from django.db import models


class Student(models.Model):
    mssv = models.CharField(max_length=13, primary_key=True)  # MSSV có độ dài 13 ký tự
    name = models.CharField(max_length=50, null=True, blank=True)
    lop = models.CharField(max_length=10, null=True, blank=True)
    khoa = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f"{self.mssv} - {self.name} - {self.lop} - {self.khoa}"
    
class Attendance(models.Model):
    mssv = models.CharField(max_length=13)  # MSSV có độ dài 13 ký tự
    time = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f"{self.mssv} - {self.time.strftime('%d/%m/%Y %H:%M:%S')}"
    
class In4SV(models.Model):  # Đặt tên bảng theo quy ước Django
    mssv = models.CharField(max_length=20, unique=True)
    ten = models.CharField(max_length=100)
    lop = models.CharField(max_length=50)
    khoa = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.mssv} - {self.ten}"

class Lich(models.Model):
    malich = models.AutoField(primary_key=True)
    lop = models.CharField(max_length=100)
    tiet = models.CharField(max_length=100)
    ngay = models.CharField(max_length=100)
    maphong = models.CharField(max_length=100, null=True)
    magv = models.CharField(max_length=100, null=True)
    
    class Meta:
        db_table = "lich"
        
    def __str__(self):
        return f"{self.lop} - {self.ngay} - {self.tiet}"

