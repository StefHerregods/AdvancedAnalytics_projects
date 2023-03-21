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

    # Return the cleaned dataframe
    return df


def drop_text(df):

    df = df.select_dtypes(include=np.number)

    return df