from rest_framework import views, status
from rest_framework.response import Response
from .models import AnalysisResult
from .serializers import AnalysisResultSerializer
from .utils import simple_sentiment_analysis


class AnalysisAPIView(views.APIView):
    def post(self, request, dataset_id):
        analysis_result = simple_sentiment_analysis(request.data)
        result_obj = AnalysisResult.objects.create(dataset_id=dataset_id, result=analysis_result)
        serializer = AnalysisResultSerializer(result_obj)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
