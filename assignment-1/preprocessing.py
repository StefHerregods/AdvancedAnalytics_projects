import pandas as pd
import numpy as np

def clean_data(df):

    # Drop unnecessary columns
    df.drop(columns=['property_sqfeet', 'host_id', 'host_since', 'property_last_updated'], inplace=True)

    # Replace all missing values in text columns with an empty string
    df.fillna({'property_name': "", 'property_summary': "", 'property_space': "", 'property_desc': "", 'property_neighborhood': "", 'property_notes': "", 'property_transit': "", 'property_access': "", 'property_interaction': "", 'property_rules': "", 'host_about': ""}, inplace=True)
    
    # Create columns with text length for all text cols
    df = df.assign(property_name_len = df['property_name'].str.len(), property_summary_len = df['property_summary'].str.len(), property_space_len = df['property_space'].str.len(), property_desc_len = df['property_desc'].str.len(), property_neighborhood_len = df['property_neighborhood'].str.len(), property_notes_len = df['property_notes'].str.len(), property_transit_len = df['property_transit'].str.len(), property_access_len = df['property_access'].str.len(), property_interaction_len = df['property_interaction'].str.len(), host_about_len = df['host_about'].str.len(), property_rules_len = df['property_rules'].str.len())

    # Create a column with the total length of all text columns
    df = df.assign(property_full_len = df['property_name_len'] + df['property_summary_len'] + df['property_space_len'] + df['property_desc_len'] + df['property_neighborhood_len'] + df['property_notes_len'] + df['property_transit_len'] + df['property_access_len'] + df['property_interaction_len'] + df['property_rules_len'] + df['host_about_len'])

    # One-hot encoding
    df = pd.get_dummies(df, columns=['property_room_type'])
    df = pd.get_dummies(df, columns=['property_type'])
    df = pd.get_dummies(df, columns=['property_bed_type'])
    df = pd.get_dummies(df, columns=['booking_cancel_policy'])

    # For 'property_amenities', replace missing rows with most common values and then multi-label encode
    df.fillna({'property_amenities': df['property_amenities'].mode()[0]}, inplace=True)
    loc_0 = df.columns.get_loc('property_amenities')
    df_encoded = df['property_amenities'].str.get_dummies(sep=', ').add_prefix('property_amenities_')
    df = pd.concat([df.iloc[:,:loc_0], df_encoded, df.iloc[:,loc_0+1:]], axis=1)

    # For 'extra', replace missing rows with most common values and then multi-label encode
    df.fillna({'extra': df['extra'].mode()[0]}, inplace=True)
    loc_0 = df.columns.get_loc('extra')
    df_encoded = df['extra'].str.get_dummies(sep=', ').add_prefix('extra_')
    df = pd.concat([df.iloc[:,:loc_0], df_encoded, df.iloc[:,loc_0+1:]], axis=1)

    # Multi-label encode 'host_verified'
    loc_0 = df.columns.get_loc('host_verified')
    df_encoded = df['host_verified'].str.get_dummies(sep=', ').add_prefix('host_verified_')
    df = pd.concat([df.iloc[:,:loc_0], df_encoded, df.iloc[:,loc_0+1:]], axis=1)

    # Drop text
    df = df.select_dtypes(include=np.number)

    # Return the cleaned dataframe
    return df


def remove_missing(df):

    # Replace missing values with the median of each column in: 'property_bathrooms', 'property_bedrooms', 'property_beds'
    df.fillna({'property_bathrooms': df['property_bathrooms'].median(), 'property_bedrooms': df['property_bedrooms'].median(), 'property_beds': df['property_beds'].median()}, inplace=True)
    
    # Drop rows with missing data in column: 'host_response_rate'
    df.dropna(subset=['host_response_rate'], inplace=True)
    
    # Drop columns: 'reviews_value', 'reviews_location' and 6 other columns
    df.drop(columns=['reviews_value', 'reviews_location', 'reviews_per_month', 'reviews_communication', 'reviews_checkin', 'reviews_cleanliness', 'reviews_acc', 'reviews_rating'], inplace=True)
    
    return df



def simple_clean(df):

    # Replace all missing values in text columns with an empty string
    df.fillna({'property_name': "", 'property_summary': "", 'property_space': "", 'property_desc': "", 'property_neighborhood': "", 'property_notes': "", 'property_transit': "", 'property_access': "", 'property_interaction': "", 'property_rules': "", 'host_about': ""}, inplace=True)

    # Create columns with text length for all text cols
    df = df.assign(property_name_len = df['property_name'].str.len(), property_summary_len = df['property_summary'].str.len(), property_space_len = df['property_space'].str.len(), property_desc_len = df['property_desc'].str.len(), property_neighborhood_len = df['property_neighborhood'].str.len(), property_notes_len = df['property_notes'].str.len(), property_transit_len = df['property_transit'].str.len(), property_access_len = df['property_access'].str.len(), property_interaction_len = df['property_interaction'].str.len(), host_about_len = df['host_about'].str.len(), property_rules_len = df['property_rules'].str.len())

    # Create a column with the total length of all text columns
    df = df.assign(property_full_len = df['property_name_len'] + df['property_summary_len'] + df['property_space_len'] + df['property_desc_len'] + df['property_neighborhood_len'] + df['property_notes_len'] + df['property_transit_len'] + df['property_access_len'] + df['property_interaction_len'] + df['property_rules_len'] + df['host_about_len'])

    # Drop columns: 'property_name', 'property_summary' and 33 other columns
    df.drop(columns=['property_zipcode', 'property_lat', 'property_lon', 'property_bed_type', 'property_amenities', 'property_sqfeet', 'property_scraped_at', 'property_last_updated', 'host_id', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_nr_listings', 'host_nr_listings_total', 'host_verified', 'booking_price_covers', 'booking_availability_30', 'booking_availability_60', 'booking_availability_90', 'booking_availability_365', 'reviews_first', 'reviews_last', 'extra'], inplace=True)
    
    # Drop rows with missing data in columns: 'reviews_rating', 'reviews_acc' and 6 other columns
    df.dropna(subset=['reviews_rating', 'reviews_acc', 'reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value', 'reviews_per_month'], inplace=True)
    
    # Replace missing values with the median of each column in: 'property_bathrooms', 'property_bedrooms', 'property_beds'
    df.fillna({'property_bathrooms': df['property_bathrooms'].median(), 'property_bedrooms': df['property_bedrooms'].median(), 'property_beds': df['property_beds'].median()}, inplace=True)
    
    # Encode ordinal column
    df['booking_cancel_policy'] = df['booking_cancel_policy'].replace({'flexible':1, 'moderate':2, 'strict':3, 'super_strict_30':4})
    
    # One-hot encode columns: 'property_type', 'property_room_type'
    df = pd.get_dummies(df, columns=['property_type', 'property_room_type'])

    # Scale columns 'property_max_guests', 'property_bathrooms' and 13 other columns between 0 and 1
    new_min, new_max = 0, 1
    old_min, old_max = df['property_max_guests'].min(), df['property_max_guests'].max()
    df['property_max_guests'] = (df['property_max_guests'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['property_bathrooms'].min(), df['property_bathrooms'].max()
    df['property_bathrooms'] = (df['property_bathrooms'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['property_bedrooms'].min(), df['property_bedrooms'].max()
    df['property_bedrooms'] = (df['property_bedrooms'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['property_beds'].min(), df['property_beds'].max()
    df['property_beds'] = (df['property_beds'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['booking_min_nights'].min(), df['booking_min_nights'].max()
    df['booking_min_nights'] = (df['booking_min_nights'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['booking_max_nights'].min(), df['booking_max_nights'].max()
    df['booking_max_nights'] = (df['booking_max_nights'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_num'].min(), df['reviews_num'].max()
    df['reviews_num'] = (df['reviews_num'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_rating'].min(), df['reviews_rating'].max()
    df['reviews_rating'] = (df['reviews_rating'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_acc'].min(), df['reviews_acc'].max()
    df['reviews_acc'] = (df['reviews_acc'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_cleanliness'].min(), df['reviews_cleanliness'].max()
    df['reviews_cleanliness'] = (df['reviews_cleanliness'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_checkin'].min(), df['reviews_checkin'].max()
    df['reviews_checkin'] = (df['reviews_checkin'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_communication'].min(), df['reviews_communication'].max()
    df['reviews_communication'] = (df['reviews_communication'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_location'].min(), df['reviews_location'].max()
    df['reviews_location'] = (df['reviews_location'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_value'].min(), df['reviews_value'].max()
    df['reviews_value'] = (df['reviews_value'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    old_min, old_max = df['reviews_per_month'].min(), df['reviews_per_month'].max()
    df['reviews_per_month'] = (df['reviews_per_month'] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    # Drop text
    df = df.select_dtypes(include=np.number)

    return df
