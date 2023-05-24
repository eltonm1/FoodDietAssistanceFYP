from django.db import models
from user_manager import models as user_models
from django.utils import timezone
import uuid
# Create your models here.


class NutritionInformation(models.Model):
    energy = models.IntegerField(blank=False)
    protein = models.FloatField(blank=False)
    total_fat = models.FloatField(blank=False)
    saturated_fat = models.FloatField(blank=False)
    trans_fat = models.FloatField(blank=False)
    carbohydrates = models.FloatField(blank=False)
    sugars = models.FloatField(blank=False)
    sodium = models.FloatField(blank=False)

    cholesterol = models.FloatField(null=True, blank=True)
    vitaminB2 = models.FloatField(null=True, blank=True)
    vitaminB3 = models.FloatField(null=True, blank=True)
    vitaminB6 = models.FloatField(null=True, blank=True)

class ProductPrice(models.Model):
    SUPERMARKET_CHOICES = (
        ('Wellcome', 'Wellcome'),
        ('Parknshop', 'Parknshop'),
        ('Jasons', 'Jasons'),
        ('Aeon', 'Aeon'),
        ('Dchfood', 'Dchfood'),
        ('Watsons', 'Watsons'),
        ('Mannings', 'Mannings'),
        ('Ztore', 'Ztore'),
    )
    price = models.FloatField(blank=False)
    supermarket = models.CharField(max_length=10, blank=False, choices=SUPERMARKET_CHOICES, default='1Wellcome')
    date = models.DateTimeField(default=timezone.now)

class FoodProduct(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=True)
    # id = models.CharField(max_length=300, blank=False, primary_key=True)
    name = models.CharField(max_length=300, blank=False)
    category_1 = models.CharField(max_length=100, blank=False, null=True)
    category_2 = models.CharField(max_length=100, blank=False, null=True)
    category_3 = models.CharField(max_length=100, blank=False, null=True)
    barcode = models.CharField(max_length=300, blank=True)
    pricewatchcode = models.CharField(max_length=16, blank=True, unique=True)
    created = models.DateTimeField(default=timezone.now)
    nutrition = models.ForeignKey(NutritionInformation, on_delete=models.CASCADE, null=True, blank=True)
    manufacturer = models.CharField(max_length=200, blank=True)
    brand = models.CharField(max_length=200, blank=True)
    product_price = models.ManyToManyField(ProductPrice, blank=True)

    def latest_pricing(self):
        return self.product_price.order_by("-date")[:10]
