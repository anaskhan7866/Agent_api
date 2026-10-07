from django.urls import path
from .views import QueryRAGView, ChatInterfaceView, UploadPDFView

urlpatterns = [
    path('', ChatInterfaceView.as_view(), name='chat-interface'),
    path('query/', QueryRAGView.as_view(), name='query-rag'),
    path('upload/', UploadPDFView.as_view(), name='upload-pdf'),
]
