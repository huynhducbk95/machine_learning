
from django.conf.urls import url
from . import views

urlpatterns = [
    url(u'^$',views.index,name='index'),
    url(u'^predict$',views.predict,name='predict'),
]