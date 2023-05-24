from django.contrib.auth.models import Group
from user_manager.models import User
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    date_of_birth = serializers.DateField(format="%Y-%m-%dT%H:%M:%SZ")
    user_avator = serializers.SerializerMethodField()
    class Meta:
        model = User
        fields = ['email', 'last_name' , 'first_name', 'date_of_birth', 'groups', 'gender', 'user_avator']
   
    def get_user_avator(self, obj: User):
        request = self.context.get('request')
        
        if obj.user_avatar:
            photo_url = obj.user_avatar.url
            return request.build_absolute_uri(photo_url)
        else:
            return None
            
class CreateUserSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ('email', 'password', 'date_of_birth', 'last_name', 'first_name', 'gender')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        print("CREATING", validated_data)
        password = validated_data.pop('password')
        print(password)
        user = User.objects.create_user(
                **validated_data,
        )
        user.set_password(password)
        user.save()
        return user

