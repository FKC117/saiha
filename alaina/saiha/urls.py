from django.urls import path
from saiha import views

app_name = "saiha"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload-dataset/", views.upload_dataset, name="upload_dataset"),
    path("api/chat-analysis/", views.api_chat_analysis, name="api_chat_analysis"),
    path("update-profile/", views.update_profile, name="update_profile"),
    path("datasets/", views.dataset_dashboard, name="dataset_dashboard"),
    path("datasets/delete/<uuid:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("datasets/<uuid:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("usage/stats/", views.get_usage_data, name="get_usage_data"),
    path("api/analysis-result/<uuid:result_id>/", views.get_analysis_result, name="get_analysis_result"),
    path("api/export/session/<uuid:session_id>/<str:format>/", views.export_session_report, name="export_session_report"),

    # Corporate Admin Panel
    path("corporate/login/", views.corporate_login, name="corporate_login"),
    path("corporate/join/<uuid:token>/", views.corporate_join, name="corporate_join"),
    path("user/topup/", views.user_topup, name="user_topup"),
    path("corporate/dashboard/", views.corporate_dashboard, name="corporate_dashboard"),
    path("corporate/analytics/", views.corporate_analytics, name="corporate_analytics"),
    path("corporate/api/usage/", views.get_corporate_usage_data, name="get_corporate_usage_data"),
    path("corporate/members/add/", views.corporate_add_member, name="corporate_add_member"),
    path("corporate/members/remove/", views.corporate_remove_member, name="corporate_remove_member"),
    path("corporate/members/resend/", views.corporate_resend_invite, name="corporate_resend_invite"),
    path("corporate/members/reallocate/", views.corporate_reallocate_credits, name="corporate_reallocate_credits"),
    path("corporate/members/purchase-seats/", views.corporate_purchase_seats, name="corporate_purchase_seats"),
    path("corporate/topup/", views.corporate_topup, name="corporate_topup"),
    path("corporate/simulate-recharge/", views.simulate_corporate_recharge, name="simulate_corporate_recharge"),

    # Billing & Invoices
    path("billing/invoice/<uuid:invoice_id>/", views.invoice_detail, name="invoice_detail"),
    path("api/billing/resend/", views.api_resend_invoice, name="api_resend_invoice"),
    path("api/billing/history/", views.api_billing_history, name="api_billing_history"),
]
