import pandas as pd
import numpy as np
import os
from env import host, user, password

#function uses info from env.py file to create a connection url to access db

def get_connection(db, user=user, host=host, password=password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#titanic data set func
def new_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    #sql query
    df = pd.read_sql(sql_query, get_connection('titanic_db')) 
    #read in data frame
    return df

#reads titanic data from codeup db, writes to a csv file if it doesnt exist and returns a df
def get_titanic_data():
    
    if os.path.isfile('titanic_df.csv'):
        
        df = pd.read_csv('titanic_df.csv', index_col=0)
        #if csv file exists read data in from csv
    
    else:
    #read fresh data from db into df
        
        df = new_titanic_data()
    #wrote df to csv
        
        df.to_csv('titanic_df.csv')
    
    return df

#reads iris data
def new_iris_data():
    
    sql_query = """
                SELECT
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING (species_id)
                """
    
    #read in df from codeup db
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df

#reads in iris data from db, writes data to csv if no local file exists, returns a df
def get_iris_data():
    
    if os.path.isfile('iris_df.csv'):
        #if csv exists read in data
        df = pd.read_csv('iris_df.csv', index_col=0)
    
    else:
        #read in fresh data from db to df
        df = new_iris_data()
        #cache data
        df.to_csv('iris_df.csv')
        
    return df

##reads telco data 
def new_telco_data():
    
    sql_query = """
                SELECT * FROM customers
                JOIN contract_types using (contract_type_id)
                JOIN internet_service_types using (internet_service_type_id)
                JOIN payment_types using (payment_type_id)
                """
    
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df

##reads in telco data from db, writed data to csv in no local file exists, returns a df
def get_telco_data():
    
    if os.path.isfile('telco.csv'):
        #if csv exists read in data
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        #read in fresh data from db to df
        df = new_telco_data()
        #cache data
        df.to_csv('telco.csv')
        
    return df