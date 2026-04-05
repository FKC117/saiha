from django.urls import path
from . import views

app_name = "saiha"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload-dataset/", views.upload_dataset, name="upload_dataset"),
    path("api/quanta-chat/", views.api_quanta_chat, name="api_quanta_chat"),
    path("update-profile/", views.update_profile, name="update_profile"),
    path("datasets/", views.dataset_dashboard, name="dataset_dashboard"),
    path("datasets/delete/<uuid:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("datasets/<uuid:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("api/analysis-result/<uuid:result_id>/", views.get_analysis_result, name="get_analysis_result"),
    path("api/export/session/<uuid:session_id>/<str:format>/", views.export_session_report, name="export_session_report"),
]
