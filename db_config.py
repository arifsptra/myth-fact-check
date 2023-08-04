import mysql.connector

class Db_Config:
    def __init__(self, host, username, pwd, dbName, table):
        self.host = host
        self.username = username
        self.pwd = pwd
        self.dbName = dbName
        self.table = table

    def fetchAll(self):
        db_config = {
            "host": str(self.host),
            "user": str(self.username),
            "password": str(self.pwd),
            "database": str(self.dbName),
        }

        try:
            connection = mysql.connector.connect(**db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                table_name = str(self.table)
                select_query = f"SELECT pernyataan, target FROM {table_name};"
                cursor.execute(select_query)
                rows = cursor.fetchall()
                return rows
        except mysql.connector.Error as e:
            return "Error while connecting to MySQL:" + e
            