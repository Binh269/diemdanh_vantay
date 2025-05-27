from django.apps import AppConfig
from django.db import connection



class ShowConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Show'
    def ready(self):
        """Xóa toàn bộ session khi server restart"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM django_session;")
            print("🔥 Server restart - Đã xóa tất cả session!")
        except Exception as e:
            print(f"⚠️ Lỗi khi xóa session: {e}")


