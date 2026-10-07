from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag_service import query_rag, ingest_single_pdf
import os
from django.conf import settings
from rest_framework.parsers import MultiPartParser
from django.views.generic import TemplateView

class QueryRAGView(APIView):
    def post(self, request, *args, **kwargs):
        query = request.data.get('query')
        if not query:
            return Response({"error": "Please provide a 'query' in the request body."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = query_rag(query)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatInterfaceView(TemplateView):
    template_name = 'agent_api/chat.html'


class UploadPDFView(APIView):
    parser_classes = [MultiPartParser]
    
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "Please provide a 'file' parameter."}, status=status.HTTP_400_BAD_REQUEST)
        if not file_obj.name.lower().endswith('.pdf'):
            return Response({"error": "Only PDF files are supported."}, status=status.HTTP_400_BAD_REQUEST)
            
        raw_docs_dir = os.path.join(settings.BASE_DIR, 'data', 'raw_docs')
        os.makedirs(raw_docs_dir, exist_ok=True)
        file_path = os.path.join(raw_docs_dir, file_obj.name)
        
        with open(file_path, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
                
        try:
            chunks_processed = ingest_single_pdf(file_path)
            return Response({"message": f"Successfully ingested {file_obj.name} ({chunks_processed} chunks added)."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
