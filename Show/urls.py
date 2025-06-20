from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("register/", register, name="register"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("update_profile/", update_profile, name="update_profile"),
    path('check-session/', check_session, name='check_session'),
    path('get-random-token/', get_random_token, name='get_random_token'),
    # path('', index, name='index'),
    # path('phong', Phong, name='phong'),
    # path('phong_Delete/<int:id>/', Phong_Delete, name='phong_Delete'),
    # path('phong_Update/<str:id>', Phong_Update, name='phong_Update'),
    # path('thietbi_Delete/<int:id>/', ThietBi_Delete, name='thietbi_Delete'),
    # path('thietbi_Update/<str:id>', ThietBi_Update, name='thietbi_Update'),
    path('ql_sv/', ql_sv, name='ql_sv'),
    path('lich_view/', lich_view, name='lich_view'),
    path('get_weekly_schedule', get_weekly_schedule, name='get_weekly_schedule'),
    path('get_members/', get_members, name='get_members'),
    path('add_student_to_class/', add_student_to_class, name='add_student_to_class'),
    path('update_student_in_class/', update_student_in_class, name='update_student_in_class'),
    path('delete_student_from_class/', delete_student_from_class, name='delete_student_from_class'),
    path('import-students-to-class-excel/', import_students_to_class_excel, name='import_students_to_class_excel'),
    path('export-students-by-class/', export_students_by_class, name='export_students_by_class'),
    path('import-excel/', import_excel, name='import_excel'),
    path('export-excel/', export_excel, name='export_excel'),
    path('diemdanh/', Diemdanh, name='diemdanh'),
    path('get_attendance_history/', get_attendance_history, name='get_attendance_history'),
    path('get_available_classes/', get_available_classes, name='get_available_classes'),
    path('mark_attendance/', diemdanh, name='mark_attendance'),
    path('get_class_members/', get_class_members, name='get_class_members'),
    path('manual_attendance/', manual_attendance, name='manual_attendance'),
    path('them_sv/', them_sv, name='them_sv'),
    path('add_folder/', add_folder, name='add_folder'),
    path("", thongke_view, name="thongke"),
    path('thongke_chi_tiet/', thongke_chi_tiet, name='thongke_chi_tiet'),
    path('export_thongke_excel/', export_thongke_excel, name='export_thongke_excel'),
    path("manual-open/", manual_open, name="manual_open"),
    path('fingerprint/register/', register_fingerprint, name='register_fingerprint'),
    path('fingerprint/attendance/', diemdanh_vantay, name='diemdanh_vantay'),
    path('fingerprint/sensor/', check_fingerprint_sensor, name='check_fingerprint_sensor'),
    path('fingerprint/enroll/', enroll_fingerprint, name='enroll_fingerprint'),
    path('settings_view/', settings_view, name='settings_view'),
    path('get_user_settings/', get_user_settings, name='get_user_settings'),
    path('save_settings/', save_settings, name='save_settings'),
    path('setting/', setting, name='setting'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
