from .models import FoodProduct, ProductPrice, NutritionInformation
from user_manager.models import User
from rest_framework import serializers
import user_manager.serializer as user_serializer
from collections import OrderedDict


class NutritionInformationSerializer(serializers.ModelSerializer):
    class Meta:
        model = NutritionInformation
        fields = '__all__'

class ProductPriceSerializer(serializers.ModelSerializer):
    date = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%SZ", required=False)

    class Meta:
        model = ProductPrice
        fields = ['price', 'supermarket', 'date']

class FoodProductsSerializer(serializers.ModelSerializer):
    product_price = ProductPriceSerializer(many=True)#, source="latest_pricing")
    nutrition = NutritionInformationSerializer()
    created = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%SZ")
    class Meta:
        model = FoodProduct
        fields = ['id', 
                    'name', 
                    'barcode',
                    'created',
                    'manufacturer',
                    'brand',
                    'product_price',
                    'nutrition',
                    'category_1',
                    'category_2',
                    'category_3'
                 ]

    def to_representation(self, instance):
        result = super(FoodProductsSerializer, self).to_representation(instance)
        return OrderedDict([(key, result[key]) for key in result if result[key] is not None])

class FoodProductsPostSerializer(serializers.ModelSerializer):

    product_price = ProductPriceSerializer(many=True, allow_null=True)
    nutrition = NutritionInformationSerializer(allow_null=True)

    class Meta:
        model = FoodProduct
        fields = [
                    'id',
                    'name', 
                    'barcode',
                    # 'created',
                    'manufacturer',
                    'brand',
                    'product_price',
                    'nutrition',
                 ]

    def create(self, validated_data):
        product_prices = validated_data.pop('product_price')
        nutrition = validated_data.pop('nutrition')

        product_price_save = []
        
        if product_prices is not None:
            for product_price in product_prices:
                product_price = ProductPrice.objects.create(**product_price)
                product_price_save.append(product_price)
        
        instance = FoodProduct.objects.create(**validated_data)
        instance.product_price.set(product_price_save)

        if nutrition is not None:
            nutrition = NutritionInformation.objects.create(**nutrition)
            instance.nutrition = nutrition

        instance.save()
        return instance 

    def update(self, instance, validated_data):
        nutrition = validated_data.pop('nutrition')
        print("**********")
        print(validated_data)
        print("**********")
        if nutrition is not None:
            nutrition = NutritionInformation.objects.create(**nutrition)
            instance.nutrition = nutrition

        instance.save()
        return instance          