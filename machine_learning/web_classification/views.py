from django.shortcuts import render
from django.http import HttpResponse
from classification.main import excute
import json
import os.path

CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


def index(request):
    return render(request, 'base.html', {})


def predict(request):
    result = {
        'result': 'not found',
    }
    if (request.method == 'POST'):
        alogrithm = request.POST.get('alogrithm', None)
        content = request.POST.get('content', None)
        text_extraction = request.POST.get('text_extraction', None)
        tag = excute(alogrithm,text_extraction,content)

        result['result'] = tag
    return HttpResponse(json.dumps(result), content_type='application/json')
