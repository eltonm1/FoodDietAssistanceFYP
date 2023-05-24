from django.contrib import admin
from .models import FoodProduct, ProductPrice, NutritionInformation

# Register your models here.

class CustomFoodProductAdmin(admin.ModelAdmin):
    # ...
    ordering = ('name', '-created')
    list_display = ('name', 'brand', 'manufacturer', 'created')
    search_fields = ('name', 'created')
    filter_horizontal = ()
    list_filter = ('brand', 'manufacturer')
    fieldsets = ()
    raw_id_fields = ("product_price",)


admin.site.register(FoodProduct, CustomFoodProductAdmin)
admin.site.register(ProductPrice)
admin.site.register(NutritionInformation)