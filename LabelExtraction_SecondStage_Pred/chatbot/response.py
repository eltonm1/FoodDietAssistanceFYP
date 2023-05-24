import random
import requests
import pickle

BACKEND_URL = "http://20.205.60.6:8000"
# BACKEND_URL = "http://localhost:8000"

def product_search(prod_entity, days):
    url = f"{BACKEND_URL}/api/foodproducts/search/{prod_entity}"
    if days:
        url += f"/{days}"
    prod_search_result = requests.get(url)
    prod_search_result = prod_search_result.json()
    return prod_search_result

def get_prod_by_ids(ids, days=1):
    url = f"{BACKEND_URL}/api/foodproducts/pkey/{'&'.join(ids)}/{days}"
    prod_get_result = requests.get(url)
    prod_get_result = prod_get_result.json()
    return prod_get_result

def greeting(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))
    days = None

    response = "Hi! I am a chatbot. What are you looking for?"

    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": greeting.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": [],
    }
    return ret_content

def what_can_you_do(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))
    days = None
    response = "Hi! I am a chatbot. I can help you find the price and details of food products."
    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": what_can_you_do.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": [],
    }
    return ret_content

def undefined_intent(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))
    days = None
    response = "I dont understand your question. Please ask me something else."
    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": undefined_intent.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": [],
    }
    return ret_content

def product_price(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))
    days = None

    products = []
    def post_process_products(prod_search_result):
        if len(prod_search_result) == 1:
            prod = prod_search_result[0]
            min_price = min([p['price'] for p in prod['product_price']])
            max_price = max([p['price'] for p in prod['product_price']])
            num_supermarket = len({p['supermarket'] for p in prod['product_price']})
            if days == 1:
                if min_price == max_price and num_supermarket > 1:
                    response = f"The price of {prod['name']} is ${min_price} across {num_supermarket} supermarkets."
                elif min_price == max_price and num_supermarket == 1:
                    response = f"The price of {prod['name']} is ${min_price} at 1 supermarket."
                else:
                    response = f"The price of {prod_entity} ranges from ${min_price} to ${max_price} across {num_supermarket} supermarkets."
            else:
                if min_price == max_price:
                    response = f"The price of {prod['name']} was ${min_price} {date_entity}."
                else:
                    response = f"The price of {prod_entity} was ${min_price} - ${max_price} {date_entity}."
        elif not prod_search_result:
            response = "Sorry, I could not find any products with that name."
        else:
            min_price = []
            max_price = []
            supermarket = set()

            for prod in prod_search_result:
                if prod['product_price']:
                    tmp_min_price = min([p['price'] for p in prod['product_price']])
                    tmp_max_price = max([p['price'] for p in prod['product_price']])
                    tmp_supermarket = {p['supermarket'] for p in prod['product_price']}
                    supermarket.update(tmp_supermarket)
                    min_price.append(tmp_min_price)
                    max_price.append(tmp_max_price)
            if min_price:
                min_price = min(min_price)
                max_price = max(max_price)
                num_supermarket = len(supermarket)
                response = f"I have found the price of {len(prod_search_result)} products with the name {prod_entity} ranging from ${min_price} to ${max_price} across {num_supermarket} supermarkets. You can have a look below."
            else:
                response = f"Something went wrong in finding the price."
            
        return response

    #Decode date entity
    if(date_entity):
        if date_entity in ["yesterday", "today"]:
            days = 1
        elif date_entity == "last week":
            days = 7
        elif date_entity == "last month":
            days = 30
    else:
        days = 1

    #Situation where the user use "it" or "that" or "this" to refer to the product
    if prod_entity in ["it", "that", "this", '']:
        #If there is a previous product context
        if prev_prod_entity: 
            prod_entity = prev_prod_entity
            prod_search_result = product_search(prod_entity, days)
            response = post_process_products(prod_search_result)
            products = prod_search_result
        else:
            response = random.choice(["Sorry, I don't know which product you are referring to. Please provide the product name.",
                "Sorry, I don't have any context of the product. Please provide the product name.",
                "Sorry, I don't know what product you are referring to. Please try again by providing a product name this time.",
            ])
    else:
        #Product search
        if prod_entity:
            prod_search_result = product_search(prod_entity, days)
            response = post_process_products(prod_search_result)
            products = prod_search_result
        else:
            response = "Sorry, I don't know what product you are referring to. Please try again by providing a product name this time."
    print(response)
    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": product_price.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": products,
    }
    return ret_content

def product_details(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))


    if prod_entity in ["it", "that", "this", '']:
        #If there is a previous product context
        if prev_prod_entity: 
            prod_entity = prev_prod_entity

    prod_search_result = product_search(prod_entity, 10) if prod_entity else []

    if prod_search_result:
        num_prod = len(prod_search_result)
        response = f"I have found {num_prod} product{'s' if num_prod > 1 else ''} with the name {prod_entity}. You can have a look below."
    else:
        if prod_entity:
            response = f"Sorry, I could not find any products with the name {prod_entity}."
        else:
            response = "Sorry, I don't know what product you are referring to. Please try again by providing a product name this time."

    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": product_details.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": prod_search_result,
    }
    return ret_content

def where_to_buy_product(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))


    if prod_entity in ["it", "that", "this", '']:
        #If there is a previous product context
        if prev_prod_entity: 
            prod_entity = prev_prod_entity

    prod_search_result = product_search(prod_entity, 10) if prod_entity else []

    if prod_search_result:
        num_prod = len(prod_search_result)
        if num_prod == 1:
            response = f"{prod_entity} is available at {', '.join({p['supermarket'] for p in prod_search_result[0]['product_price']})}."
        else:
            sueprmarkets = set()
            for product in prod_search_result:
                for product_price in product['product_price']:
                    sueprmarkets.add(product_price['supermarket'])  
            response = f"I have found {num_prod} product{'s' if num_prod > 1 else ''} with the name {prod_entity}. They are available to buy at {', '.join(sueprmarkets)}."

    else:
        if prod_entity:
            response = f"Sorry, I could not find any products with the name {prod_entity}."
        else:
            response = "Sorry, I don't know what product you are referring to. Please try again by providing a product name this time."

    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": where_to_buy_product.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": prod_search_result,
    }
    return ret_content


product_simil = pickle.load(open("chatbot/product_similarity.pickle", "rb"))
def find_similar_product(**kwargs):
    client_input = kwargs.get('client_input')
    prev_prod_entity = kwargs.get('prev_prod_entity')
    intent_id = kwargs.get('intent_id')
    prod_entity = kwargs.get('prod_entity')
    date_entity = ' '.join(kwargs.get('date_entity', []))
    
    prod_return = []
    if prod_entity in ["it", "that", "this", '']:
        #If there is a previous product context
        if prev_prod_entity: 
            prod_entity = prev_prod_entity

    prod_search_result = product_search(prod_entity, 10) if prod_entity else []
    prod_return = prod_search_result

    if prod_search_result:
        sim_prod_ids = set()
        for product in prod_search_result:
            prod_id = product['id']
            sim_products_tuple = product_simil.get(prod_id)
            if sim_products_tuple:
                sim_products_tuple = sim_products_tuple[:10]
                sim_products_id = [id for _, id in sim_products_tuple]
                sim_prod_ids.update(sim_products_id)
        prod_return = get_prod_by_ids(sim_prod_ids, 1)
        response = f"Here are some of the similar products to {prod_entity}."

    else:
        if prod_entity:
            response = f"Sorry, I could not find any products with the name {prod_entity}. Wont be able to find similar products."
        else:
            response = "Sorry, I don't know what product you are referring to. Please try again by providing a product name this time."

    ret_content = {
        "client_input": client_input,
        "intent_id": intent_id,
        "intent_str": find_similar_product.__name__,
        "prod_entity": prod_entity,
        "date_entity": date_entity,

        "response": response,
        "products_response": prod_return,
    }
    return ret_content
