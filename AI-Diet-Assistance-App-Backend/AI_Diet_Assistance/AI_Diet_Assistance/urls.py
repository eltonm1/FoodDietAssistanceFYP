"""AI_Diet_Assistance URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from rest_framework.authtoken import views
from rest_framework import routers
from user_manager.views import UserViewList, UserView, CreateUser, ProfilePicUploadView
from food_products.views import FoodProductsViewList, FoodProductsSearchViewList, FoodProductsPkeyViewList, FoodProductsSimilarityViewList
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    
    #path('api/users/<int:pk>/', UserView.as_view()),
    path('api/userprofile/', UserView.as_view()),
    path('api/users/create/', CreateUser.as_view()),
    path('api/users/', UserViewList.as_view()),
    path('api/foodproducts/', FoodProductsViewList.as_view()),
    path('api/foodproducts/<str:bcode>', FoodProductsViewList.as_view()),
    path('api/foodproducts/search/<str:query>', FoodProductsSearchViewList.as_view()),
    path('api/foodproducts/search/<str:query>/<int:days>', FoodProductsSearchViewList.as_view()),
    path('api/foodproducts/pkey/<str:id>', FoodProductsPkeyViewList.as_view()),
    path('api/foodproducts/pkey/<str:id>/<int:days>', FoodProductsPkeyViewList.as_view()),
    path('api/foodproducts/similarity/pkey/<str:id>', FoodProductsSimilarityViewList.as_view()),
    path('api/foodproducts/similarity/pkey/<str:id>/<int:days>', FoodProductsSimilarityViewList.as_view()),
    #path('api-token-auth/', views.obtain_auth_token, name='api-token-auth'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('admin/', admin.site.urls),
    # path('__debug__/', include('debug_toolbar.urls')),
    re_path(r'^upload/(?P<filename>[^/]+)$', ProfilePicUploadView.as_view()),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
