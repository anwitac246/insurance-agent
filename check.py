from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URL"))

db = client["car_insurance_mas"]

customers = db["Customer_Profiles"].find(
    {},
    {
        "_id": 0,
        "customer_id": 1,
        "full_name": 1,
        "risk_rating": 1
    }
)

print("\nCustomers:\n")

for i, c in enumerate(customers, 1):
    print(
        f"{i}. "
        f"{c.get('full_name')} | "
        f"ID: {c.get('customer_id')} | "
        f"Risk: {c.get('risk_rating')}"
    )