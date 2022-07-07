import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acquire

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import host, user, password



def clean_iris(df):
    """
    clean_iris will take an acquired df and 
    remove `species_id` and `measurement_id` columns and 
    rename `species_name` column to just `species` and
    encode 'species_name' column into TWO new columns
    
    return: single cleaned dataframe
    """
    
    dropcols = ['species_id', 'measurement_id']
    df.drop(columns= dropcols, inplace=True)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    dummy_sp = pd.get_dummies(df[['species']], drop_first=True)
    return pd.concat([df, dummy_sp], axis =1)


def prep_iris(df):
    """
    prep_iris will take one argument(df) and 
    run clean_iris to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """
    iris_df = clean_iris(df)
    train_validate, test = train_test_split(iris_df, test_size=0.2, random_state=3210, stratify=iris_df.species)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210, stratify=train_validate.species)
    return train, validate, test


def handle_missing_values(df):
    return df.assign(
        embark_town=df.embark_town.fillna('Other'),
        embarked=df.embarked.fillna('O'),
    )

def remove_columns(df):
    return df.drop(columns=['deck'])

def encode_embarked(df):
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

def prep_titanic_data(df):
    df = df\
        .pipe(handle_missing_values)\
        .pipe(remove_columns)\
        .pipe(encode_embarked)
    return df

def train_validate_test_split(df, seed=3210):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.survived
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.survived,
    )
    return train, validate, test



def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name for the stratify_by argument
    """

    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])

    return train, validate, test



def clean_titanic(df):
    """
    clean_titanic will take an acquired df and 
    covert sex column to boolean 'is_female' col
    encode "embarked" & "class" columns & add them to the end
    and drop 'age' & 'deck' cols due to NaNs
    'passenger_id' due to un-necessity
    'embark_town', 'embarked', 'sex', 'pclass', 'class' due to redundancy/enew encoded cols
    
    return: single cleaned dataframe
    """
    
    
    df["is_female"] = df.sex == "Female"
    embarked_dummies = pd.get_dummies(df.embarked, prefix='Embarked', drop_first=True)
    class_dummies = pd.get_dummies(df.pclass, prefix='class', drop_first=True)

    dropcols = ['deck', 'age', 'embark_town', 'passenger_id', 'embarked', 'sex', 'pclass', 'class']
    df.drop(columns= dropcols, inplace=True)

    return pd.concat([df, embarked_dummies, class_dummies], axis =1)

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test


def prep_telco_data(df):
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test