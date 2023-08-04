import nltk
import random
import string
import os
from db_config import Db_Config
from dotenv import load_dotenv
from function import Function
import joblib

load_dotenv()

host = os.getenv("DATABASE_HOST")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")
db_name = os.getenv("DATABASE_DB_NAME")
table = os.getenv("DATABASE_TABLE")

conn = Db_Config(host, username, password, db_name, table)
func = Function()

dataAll = conn.fetchAll()

text = "Mata kering dapat diatasi dengan istirahat reguler."

hasil = func.findResult(dataAll, text)
print(hasil)


