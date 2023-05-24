from django.shortcuts import render

# Create your views here.
from django.contrib.auth.models import Group
from user_manager.models import User
from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.http import Http404
from .serializer import UserSerializer, CreateUserSerializer
from rest_framework.parsers import FileUploadParser

class UserViewList(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = User.objects.all().order_by('-date_joined')
        serializer_context = {
            'request': request,
        }
        serializer = UserSerializer(users, context=serializer_context, many=True)
        return Response(serializer.data)


class UserView(APIView):
    permission_classes = [IsAuthenticated]

    def get_user(self, pk):
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        user = self.get_user(pk)
        serializer = UserSerializer(user, many=False)
        return Response(serializer.data)
    
    def get(self, request):
        serializer_context = {
            'request': request,
        }
        serializer = UserSerializer(request.user, context=serializer_context, many=False)
        return Response(serializer.data)

class CreateUser(APIView):
            
    def post(self, request):
        print(request.data)
        serializer = CreateUserSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)
            

class ProfilePicUploadView(APIView):
    parser_classes = [FileUploadParser]

    def put(self, request, filename, format=None):
        file_obj = request.data['file']
        user = request.user#User.objects.get(pk=3)
        user.user_avatar.save(filename, file_obj, save=True)
        serializer_context = {
            'request': request,
        }
        serializer = UserSerializer(user, context=serializer_context, many=False)
        return Response(serializer.data)