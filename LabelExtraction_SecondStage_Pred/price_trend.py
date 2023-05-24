
import requests
from bs4 import BeautifulSoup
import pandas as pd

x = pd.read_csv("price.csv")
product_code = set(x["Product Code"])

for code in product_code:
    url = f"https://online-price-watch.consumer.org.hk/opw/product/{code}"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        price = soup.find_all("div", {"class": "product-info"})
        print(price)
    except:
        print("Not found")

# Making a GET request
# r = requests.get('https://www.geeksforgeeks.org/python-programming-language/')
 
# # Parsing the HTML
# soup = BeautifulSoup(r.content, 'html.parser')
 
# # Finding by id
# s = soup.find('div', id= 'main')
 
# # Getting the leftbar
# leftbar = s.find('ul', class_='leftBarList')
 
# # All the li under the above ul
# content = leftbar.find_all('li')
 
# print(content)