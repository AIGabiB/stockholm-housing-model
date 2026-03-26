import pandas as pd
import numpy as np

def log_changes(old_shape,df,step_name):
    " Function is used to log shape before and after cleaning"
    old_rows , old_cols = old_shape
    new_rows, new_cols = df.shape

    print(f"---{step_name}---")
    print(f"Shape: {old_rows} rows, {old_cols} cols -> {new_rows} rows, {new_cols} cols")
    print(f"Rows removed: {old_rows - new_rows}")
    if old_cols >= new_cols:
        print(f"Cols removed: {old_cols - new_cols}")
    else:
        print(f"Cols added: {new_cols - old_cols}")
    print(f"----------------------")


def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    assert type(df) == pd.DataFrame, f"Data is not Pandas DataFrame, instead got {type(df)}"
    
    assert not df.empty, "File is empty"

    columns_to_drop = {"index_right": "dropped column because it was a merging index.",
                       "geometry": "dropped column because longitude and latitude were used",
                       "OrgNr": "to many unique values",
                       "gata_address": "to many unique values",
                       "deso": "to many unique values",
                       "BRF": "dropped column because of redundancy with other columns",
                       "desonamn":"to many unique values",
                       "LKF":"to many unique values",
                       "Gatuadress":"to many unique values",
                       "Utgångspris":" was not used in this project",
                       "Kr/kvm":" was not used in this project",
                       "Pris idag":" was not used in this project",
                       "Pris idag/Kvm":" was not used in this project",
                       "Gata":"to many unique values"}
    
    # Global cleaning 
    df = df.drop(columns=[c for c in columns_to_drop.keys()],errors='ignore')   
    df = df.drop_duplicates().reset_index(drop=True)

    column_mapping = {
        "Kontraktsdatum": "Contract Date",
        "Tillträdesdatum": "Move-in Date",
        "Avslutspris": "Sale Price",
        "Boarea": "Living Area (sqm)",
        "Rum": "Rooms",  
        "Våning": "Floor",
        "Våningar": "Total Floors",
        "Månavg": "Monthly Fee",
        "Årsavgift/kvm": "Annual Fee per sqm",
        "Boendeform": "Housing Type",
        "Hiss": "Elevator",
        "Balkong": "Balcony",
        "Värme ingår": "Heating Included",
        "Byggår": "Year Built",
        "Nyprod": "New Construction"
    }

    # Byt kolumnnamn
    df = df.rename(columns=column_mapping)

    return df


def clean_coordinates(df :pd.DataFrame) -> pd.DataFrame:

    # Check if Longitude columns can be safely merged
    mask_Long = df['Longitude'].notna() & df['longitude'].notna()
    check_conflicts_long = (df.loc[mask_Long, 'Longitude'] != df.loc[mask_Long, 'longitude']).sum()
    
    # Check if Latitude columns can be safely merged
    mask_Lat = df['Latitude'].notna() & df['latitude'].notna()
    check_conflicts_lat = (df.loc[mask_Lat, 'Latitude'] != df.loc[mask_Lat, 'latitude']).sum()
    
    # Merge only if safe
    if check_conflicts_long == 0 and check_conflicts_lat == 0: 
        df['long'] = df['Longitude'].fillna(df['longitude'])
        df['lat'] = df['Latitude'].fillna(df['latitude'])

        # drop redundant columns 
        df = df.drop(columns=["Longitude","longitude","Latitude","latitude"])

    else:
        print("WARNING: Coordinate conflicts detected — merge skipped.")

    # removes rows with NaN values in "long" or "lat"
    mask_long_lat_nun = (df["long"].isna()) & (df["lat"].isna())
    if len(mask_long_lat_nun) > 0:
        df = df[~mask_long_lat_nun]

    else:
        print("No NaN values found in columns 'long' or 'lat'")

    outside_stockholm_mask = (df["lat"] <= 59.2272) |  (df["lat"] >= 59.4402)  | (df["long"] >=18.2011) | (df["long"] <= 17.7605)

    df = df[~outside_stockholm_mask]

    assert (df[["long","lat"]].isna().sum() == 0 ).any() , "still NaN values in 'long','lat'"

    return df


def clean_date_cols(df :pd.DataFrame) -> pd.DataFrame:
    
    # Regex pattern for columns "Contract Date" and "Move-in Date" E.g. 2021-02-19
    reg_pattern = r'^\d{4}-\d{2}-\d{2}$'

    mask_Kontraktsdatum_regpatern = df["Contract Date"].astype(str).str.contains(reg_pattern,regex=True,na=False)
    df = df[mask_Kontraktsdatum_regpatern]

    mask_Tillträdesdatum_regpatern = df["Move-in Date"].astype(str).str.contains(reg_pattern,regex=True,na=False)
    df = df[mask_Tillträdesdatum_regpatern]

    # Change columns "Contract Date" and "Move-in Date" to datetime dtype
    df["Contract Date"] = pd.to_datetime(df["Contract Date"],errors='coerce')
    df["Move-in Date"] = pd.to_datetime(df["Move-in Date"],errors='coerce') 

    # drop row with invalid date 
    mask_invalid_datetime = df["Contract Date"] > df["Move-in Date"]
    df = df[~mask_invalid_datetime]

    df = df.drop(columns = ["Move-in Date"])

    assert (df["Contract Date"].isna().sum() == 0 ).any() , "Still NaN values in 'Contract Date'"

    return df


def clean_numerical(df :pd.DataFrame) -> pd.DataFrame:      

    numerical_columns = ['Sale Price', 'Living Area (sqm)', 'Rooms', 'Floor', 'Total Floors', 'Monthly Fee', 'Annual Fee per sqm', 'Year Built']

    # -- Floor , Total Floors -- 
    invalid_floors = (df['Floor'] > df['Total Floors']).sum()
    if invalid_floors > 0:                                                               
        df["Floor"] = np.minimum(df["Floor"],df["Total Floors"]) # # around 5000 rows were illogical where "Floor" > "Total Floors".

    # -- Year Built --
    mask_build_year = (df["Year Built"] <= 1850) & ( df["Year Built"] >= 0)
    df = df[~mask_build_year].copy()

    # -- Sale Price -- 
    mask_av = df["Sale Price"] > 30000000
    df = df[~mask_av]

    # Log-transform, because "Sale Price" was right tilted 
    df["Sale Price"] = np.log(df["Sale Price"])

    assert df[numerical_columns].isna().sum().sum() == 0 

    return df


def clean_categorical(df :pd.DataFrame) -> pd.DataFrame: 

    # values in "Housing Type" where translated to english for the users to select in the streamlit app. 
    def map_residence_type(x):
        if x in ["Radhus", "Parhus", "Kedjehus"]:
            return "House"
        elif x == "Lägenhet":
            return "Apartment"
        else:
            return "Other"

    # transform the column "Housing Type"
    df["Housing Type"] = df["Housing Type"].apply(map_residence_type)

    cols = ["Elevator", "Balcony", "Heating Included", "New Construction"]
    df[cols] = df[cols].replace({"Ja": "Yes", "Nej": "No"})

    return df


def clean_data(df:pd.DataFrame,debug: bool = False) -> pd.DataFrame:

    df = df.copy()
    
    old_shape = df.shape
    df = initial_cleaning(df)
    if debug: 
        log_changes(old_shape, df, "Initial cleaning")

    old_shape = df.shape
    df = clean_coordinates(df)
    if debug: 
        log_changes(old_shape, df, "Clean coordinates")
    
    old_shape = df.shape
    df = clean_date_cols(df)
    if debug: 
        log_changes(old_shape, df, "Clean date columns")

    old_shape = df.shape
    df = clean_numerical(df)
    if debug: 
        log_changes(old_shape, df, "Clean numerical columns")

    df = clean_categorical(df)

    return df