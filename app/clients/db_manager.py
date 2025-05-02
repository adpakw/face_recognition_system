from app.clients.postgres import PostgreSQLClient
import os
from typing import List


class DatabaseManager:
    def execute_sql_files_from_folder(self) -> None:
        """
        Чтение и выполнение всех SQL-файлов из указанной папки
        """
        folder_path = "migrations"
        if not os.path.isdir(folder_path):
            raise ValueError(f"Папка не найдена: {folder_path}")
        
        sql_files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.sql') and os.path.isfile(os.path.join(folder_path, f))]
        
        if not sql_files:
            print(f"В папке {folder_path} не найдено .sql файлов")
            return
        
        sql_files.sort()
    
        for sql_file in sql_files:
            file_path = os.path.join(folder_path, sql_file)
            print(f"Выполнение файла: {sql_file}")
            try:
                with PostgreSQLClient() as client:
                    client.execute_sql_file(file_path)
            except Exception as e:
                print(f"Ошибка при выполнении файла {sql_file}: {e}")
                raise

    