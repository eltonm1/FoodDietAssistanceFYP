# AI-Diet-Assistance-App-Backend

## First Time Setup
```
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

## Run Server
```
sudo snap start redis
python manage.py runserver 0.0.0.0:8000
celery -A food_products.celery worker -B --loglevel=info
```
