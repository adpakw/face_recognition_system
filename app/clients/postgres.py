import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import os
import pandas as pd

class PostgreSQLClient:
    def __init__(self):
        """
        Инициализация клиента PostgreSQL с параметрами из .env файла
        """
        load_dotenv(".env")
        
        self.connection_params = {
            'dbname': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT')
        }
        self.conn = None
        
    def __enter__(self):
        """Поддержка контекстного менеджера"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Поддержка контекстного менеджера"""
        self.close()
        
    def connect(self):
        """Установка соединения с БД"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
        except Exception as e:
            print(f"Ошибка подключения к PostgreSQL: {e}")
            raise
            
    def close(self):
        """Закрытие соединения с БД"""
        if self.conn is not None:
            self.conn.close()
            
    def execute_query(self, query: Union[str, sql.Composed], params: Optional[tuple] = None, 
                     return_df: bool = True) -> Optional[pd.DataFrame]:
        """
        Выполнение SQL запроса с возможностью возврата DataFrame
        
        :param query: SQL запрос
        :param params: параметры для запроса
        :param return_df: возвращать результат как DataFrame
        :return: DataFrame (если return_df=True) или None
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            if cursor.description is None: 
                self.conn.commit()
                return None
                
            if return_df:
                col_names = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=col_names)
                return df
            else:
                return cursor.fetchall()
                
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Ошибка выполнения запроса: {e}")
        finally:
            if cursor is not None:
                cursor.close()
                
            
    def insert(self, table: str, data: List[Dict[str, Any]]) -> None:
        """
        Массовая вставка записей в таблицу
        
        :param table: имя таблицы
        :param data: список словарей с данными
        """
        if not data:
            return
            
        columns = data[0].keys()
        values = [[row[col] for col in columns] for row in data]
        
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.executemany(query, values)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Ошибка массовой вставки: {e}")
            raise
        finally:
            if cursor is not None:
                cursor.close()
    
    def execute_sql_file(self, file_path: str) -> None:
        """
        Создание таблиц из SQL файла
        
        :param file_path: путь к SQL файлу
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        self.execute_query(sql_content)