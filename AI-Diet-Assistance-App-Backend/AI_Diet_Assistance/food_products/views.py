from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import FoodProduct, ProductPrice
from .serializer import FoodProductsSerializer, FoodProductsPostSerializer
# Create your views here.
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.db.models import Prefetch
from datetime import datetime, timedelta
import pytz
from django.contrib.postgres.search import SearchVector
import pickle

class FoodProductsViewList(APIView):
    # permission_classes = [HasGroupPermission]
    # permission_classes = [IsAdminUser]
    # @method_decorator(cache_page(10*1))
    # def get(self, request):
    #     foodProducts = FoodProduct.objects.all().order_by('name')
    #     serializer_context = {
    #         'request': request,
    #     }
    #     foodProducts = FoodProductsSerializer(foodProducts, context=serializer_context, many=True)

    #     return Response(foodProducts.data)
    @method_decorator(cache_page(10*1))
    def get(self, request, bcode=None):
        timelimit = datetime.now(pytz.UTC) - timedelta(days=5)
        if bcode is None:
            prefetch = Prefetch("product_price", queryset=ProductPrice.objects.filter(date__gte=timelimit).order_by('-date'))
            foodProducts = FoodProduct.objects.prefetch_related(prefetch).order_by('name').all()
            serializer_context = {
                'request': request,
            }
            foodProducts = FoodProductsSerializer(foodProducts, context=serializer_context, many=True)

            return Response(foodProducts.data)
        else:
            foodProducts = FoodProduct.objects.get(barcode=bcode)
            serializer_context = {
                'request': request,
            }
            foodProducts = FoodProductsSerializer(foodProducts, context=serializer_context, many=False)

            return Response(foodProducts.data)
        
    def post(self, request):
        print(request.data)
        serializer = FoodProductsPostSerializer(data=request.data)
        
        if serializer.is_valid(raise_exception=False):
            serializer.save()
            print(serializer.data)
            return Response(serializer.data)
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, bcode):
        print(request.data)
        food_product = FoodProduct.objects.get(barcode=bcode)
        serializer = FoodProductsPostSerializer(instance=food_product, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=False):
            serializer.save()
            print(serializer.data)
            return Response(FoodProductsSerializer(food_product, many=False, context={"request": request}).data)#serializer.data)
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FoodProductsSearchViewList(APIView):

    def get(self, request, query, days=None):
        if days is None:
            prefetch = Prefetch("product_price", queryset=ProductPrice.objects.order_by('-date'))
            foodProducts = FoodProduct.objects.annotate(search=SearchVector('name', 'brand')).filter(search=query)
        else:
            timelimit = datetime.now(pytz.UTC) - timedelta(days=days)
            prefetch = Prefetch("product_price", queryset=ProductPrice.objects.filter(date__gte=timelimit).order_by('-date'))
            foodProducts = FoodProduct.objects.annotate(search=SearchVector('name', 'brand')).filter(search=query).prefetch_related(prefetch)
        serializer_context = {
            'request': request,
        }
        foodProducts = FoodProductsSerializer(foodProducts, context=serializer_context, many=True)

        return Response(foodProducts.data)
    
class FoodProductsPkeyViewList(APIView):

    def get(self, request, id, days=None):
        id = id.split('&')
        if days is None:
            foodProducts = FoodProduct.objects.filter(id__in=id)
        else:
            timelimit = datetime.now(pytz.UTC) - timedelta(days=days)
            prefetch = Prefetch("product_price", queryset=ProductPrice.objects.filter(date__gte=timelimit).order_by('-date'))
            foodProducts = FoodProduct.objects.filter(id__in=id).prefetch_related(prefetch)
        serializer_context = {
            'request': request,
        }
        foodProducts = FoodProductsSerializer(foodProducts, context=serializer_context, many=True)

        return Response(foodProducts.data)
    
product_simil = pickle.load(open("food_products/product_similarity.pickle", "rb"))
class FoodProductsSimilarityViewList(APIView):
    def find_similar_product(self, ids):
        sim_prod_ids = set()
        for id in ids:
            sim_products_tuple = product_simil.get(id)
            if sim_products_tuple:
                sim_products_tuple = sim_products_tuple[:10]
                sim_products_id = [id for _, id in sim_products_tuple]
                sim_prod_ids.update(sim_products_id)
        return list(sim_prod_ids)
    
    def get(self, request, id, days=None):
        id = id.split('&')
        id = self.find_similar_product(id)
        if days is None:
            foodProducts = FoodProduct.objects.filter(id__in=id)
        else:
            timelimit = datetime.now(pytz.UTC) - timedelta(days=days)
            prefetch = Prefetch("product_price", queryset=ProductPrice.objects.filter(date__gte=timelimit).order_by('-date'))
            foodProducts = FoodProduct.objects.filter(id__in=id).prefetch_related(prefetch)
        serializer_context = {
            'request': request,
        }
        foodProducts = FoodProductsSerializer(foodProducts, context=serializer_context, many=True)

        return Response(foodProducts.data)

