from django.test import TestCase
from rest_framework.test import (APIRequestFactory, APIClient)
# Create your tests here.
from .models import User
from rest_framework.test import force_authenticate
from .views import UserViewList
from rest_framework_simplejwt.tokens import RefreshToken

class UserTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(
                **{
                    "email" : "test_email@test.com",
                    "date_of_birth" : "2000-01-01",
                    },
        )
        user.set_password("123")
        user.save()
        self.token = str(RefreshToken.for_user(user).access_token)
        self.user = user

    @property
    def bearer_token(self):
        # assuming there is a user in User model
        user = User.objects.get(id=1)
        refresh = RefreshToken.for_user(user)
        return refresh.access_token

    def test_get_all_users_authenticated(self):
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION='Bearer ' + self.token)
        response = client.get('/api/users/')
        self.assertEqual(response.status_code, 200)
    
    def test_get_all_users_not_authenticated(self):
        client = APIClient()
        response = client.get('/api/users/')
        self.assertEqual(response.status_code, 401)

    def test_get_current_user(self):
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION='Bearer ' + self.token)
        response = client.get('/api/userprofile/')
        self.assertEqual(response.data['email'], self.user.email)