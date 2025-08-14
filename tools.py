from typing import Dict, List, Union
from autogen_core.tools import FunctionTool
import requests  # Import requests for HTTP calls
import asyncio  # Import asyncio for async operations
from logger import logger_instance

logger_ctx = logger_instance

# Define Product and Category Data Models
class Product:
    def __init__(self, name: str, Id: int, price: float):
        self.name = name
        self.id = Id
        self.price = price

    def to_dict(self):
        return {"name": self.name, "id": self.id, "price": self.price}


class Category:
    def __init__(self, name: str):
        self.name = name
        self.products = []

    def add_product(self, product: Product):
        self.products.append(product)

    def to_dict(self):
        return {"name": self.name, "products": [product.to_dict() for product in self.products]}


#
async def get_catalog_products(search_str: str) -> List[Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]]:
    logger_ctx.info("Start get_catalog_products")
    logger_ctx.debug(f"Fetching products from catalog API related to: {search_str}")
    # Base URL for the API
    BASE_URL = 'https://dit1-digitalhealth.dxp.nttdataservices.com/umbraco/surface/ShoppingRecommendations/GetProducts'
    # Prepare parameters (if needed)
    params = {'storeID': '998b2154-0a13-43c4-8243-08dd324fbf89'}
 
    try:
        # Make a GET request to fetch the products
        response = requests.get(BASE_URL, params=params, verify=False)
        response.raise_for_status()  # Raise an error for bad responses
        # Parse the returned JSON
        json_data = response.json()  
        logger_ctx.debug(f"Catalog products fetched successfully: {json_data}")

        categories = []
        # Iterate through categories and construct Category instances
        for category in json_data:
            cat = Category(category['name'])
            for child_category in category.get('childCategories', []):  # Use get to prevent KeyErrors
                for product in child_category.get('products', []):
                    cat.add_product(Product(product['name'], product['id'], float(product['price'])))
            categories.append(cat.to_dict())  # Append converted category to the list
        logger_ctx.info("End get_catalog_products")
        return categories
    except requests.exceptions.HTTPError as http_err:
        logger_ctx.error(f'HTTP error occurred while fetching catalog products: {http_err}')
        return []  # Return empty list on error
    except Exception as err:
        logger_ctx.error(f'An error occurred while fetching catalog products: {err}')
        return []  # Return empty list on error


# Simulated Function for Fetching Products
async def get_products(search_str: str):
    print(f"Fetching products related to: {search_str}")
    # Static product catalog for demo
    electronics = Category("Electronics")
    electronics.add_product(Product("Laptop", 1, 1200.99))
    electronics.add_product(Product("Smartphone", 2, 799.99))
    electronics.add_product(Product("TV", 8, 999.99))
    electronics.add_product(Product("LG", 8, 999.99))
    electronics.add_product(Product("Samsung", 8, 999.99))
    electronics.add_product(Product("Sony", 8, 999.99))
    electronics.add_product(Product("panasonic", 8, 999.99))
    electronics.add_product(Product("Home Theater", 9, 299.99))

    furniture = Category("Furniture")
    furniture.add_product(Product("Sofa", 123, 850.00))
    furniture.add_product(Product("Seven heaven 1 sofa", 21836, 850.00))
    furniture.add_product(Product("Seven heaven 2 sofa", 21837, 850.00))
    furniture.add_product(Product("Seven heaven 3 sofa", 21838, 850.00))
    furniture.add_product(Product("Futon", 4, 175.99))
    furniture.add_product(Product("Shoe Stand", 5, 35.99))
    furniture.add_product(Product("Bed", 6, 1500.99))

    book = Category("Books")
    book.add_product(Product("Voyages",7, 19.00))

    hardware = Category("Hardware")
    hardware.add_product(Product("Hammer",11, 19.00))
    hardware.add_product(Product("Engine Oil",12, 19.00))
    hardware.add_product(Product("Bearings",13, 19.00))
    hardware.add_product(Product("Nails",14, 19.00))

    return [electronics.to_dict(), furniture.to_dict(), book.to_dict(), hardware.to_dict()]



def execute_order(product: str, price: int) -> Dict[str, Union[str, int]]:
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    return {"product":product,"price":price}
    


def look_up_item(search_query: str) -> Dict[str, str]:
    item_id = "item_132612938"
    return {"item_id":item_id,"status":"found"}


def execute_refund(item_id: str, reason: str = "not provided") -> Dict[str, str]:
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return {"item_id":item_id, "reason":reason, "refund_status":"Successful"}


execute_order_tool = FunctionTool(execute_order, description="Price should be in USD.")
look_up_item_tool = FunctionTool(
    look_up_item, description="Use to find item ID.\nSearch query can be a description or keywords."
)
execute_refund_tool = FunctionTool(execute_refund, description="")

# Define Tools for Product Finder
productfinder_tool = FunctionTool(
    get_catalog_products, description="Search the product catalog and return only filtered products.",
)
