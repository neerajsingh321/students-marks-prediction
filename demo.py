from sqlalchemy import create_engine
import pandas as pd

server = "SHUBHAM\SQLEXPRESS"
database_name = "demo"

connection_string = (

    
    f"mssql+pyodbc://@{server}/{database_name}"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

engine = create_engine(connection_string)

emp_info = pd.read_csv(r"C:\Users\Shubham\Desktop\src\employees_primary_info.csv")
emp_job_detail = pd.read_csv(r"C:\Users\Shubham\Desktop\src\employee_job_details.csv")
emp_performance = pd.read_csv(r"C:\Users\Shubham\Desktop\src\employee_performance.csv")

emp_info.to_sql(
    "EmployeesInfo",
    engine,
    
    index=False
)

emp_job_detail.to_sql(
    "EmplyeeJobDetail",
    engine,
    
    index=False
)

emp_performance.to_sql(
    "Emp_Performance",
    engine,
    
    index=False
)

print("loaded successfully")

