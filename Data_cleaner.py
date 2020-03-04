import statistics
import pandas as pd
import numpy as np
import scipy
from math import cos, asin, sqrt


class cleaners:
    
    def house_price_cleaner(self, df):
        
        self.df = df
        
        # Drop EASE-MENT column since it is empty, and UNNAMED: 0
        del df["EASE-MENT"]
        del df["Unnamed: 0"]
        
        # Change columns' names in order to have lower case and no spasce between words
        df = df.rename(columns = str.lower)
        df.columns = df.columns.str.replace(' ', '_')

        # Check for and eventually remove rows that are duplicates

        if sum(df.duplicated(df.columns)) > 0:
            df = df.drop_duplicates(df.columns, keep = "last")
    
        # sum(df_price.duplicated(df_price.columns))   >>> a possible additional check

        ## Columns that are made only of strings with no misleading values:
        # "borough", "neighborhood" and "building_class_category", "address", "apartment_number", "building_class_at_time_of_sale",
        #"sale_date"

        ## Columns that are made only of integers with no misleading values
        # "block", "lot", "zip_code", "residential_units", "commercial_units", "total_units", "tax_class_at_time_of_sale"

        # Replace a precise value in this column, which we are sure is not in the column, then drop those rows
        # >>>> df["sale_price"] = df["sale_price"].replace(' -  ', "%%%%")

        ######## ARE THOSE LINES TO BE DROPPED OR TO ADD THE MEAN
        df.drop(df[df.sale_price == ' -  '].index, inplace = True)

        # Convert all the remaining values to float
        df['sale_price'] = pd.to_numeric(df['sale_price'], errors = 'raise')

        # Convert years into integers
        
        # Replace given string with nan
        df['tax_class_at_present'] = df['tax_class_at_present'].replace(" " , np.nan)
        df['building_class_at_present'] = df['building_class_at_present'].replace(" " , np.nan)
        df['year_built'] = df['year_built'].replace(0 , np.nan)

        # The two following columns which NaN have been filled with their trimmed means
        df['land_square_feet'] = df['land_square_feet'].replace(' -  ' , 0)
        df['land_square_feet'] = df['land_square_feet'].replace('0' , 0)
        df['land_square_feet']= pd.to_numeric(df['land_square_feet'], errors = 'raise')
        df['land_square_feet'] = df['land_square_feet'].replace(0, round(scipy.stats.trim_mean(df['land_square_feet'],0.011), 0))

        df['gross_square_feet'] = df['gross_square_feet'].replace(' -  ' , 0)
        df['gross_square_feet'] = df['gross_square_feet'].replace('0' , 0)
        df['gross_square_feet']= pd.to_numeric(df['gross_square_feet'], errors = 'raise')
        df['gross_square_feet'] = df['gross_square_feet'].replace(0,round(scipy.stats.trim_mean(df['gross_square_feet'], 0.08), 0))
        # reset index 
        df = df.reset_index(drop = True) 
        return df
    
    def airbnb_cleaner(self, df):
        self.df = df
        
        del df["last_review"]
        del df["reviews_per_month"]
        df = df.rename(columns = {"neighbourhood_group": "borough", "neighbourhood": "neighborhood"}, errors = "raise")
        df = df.dropna()
        
        shape_before = df.shape
        
        # To check that every entry belongs to NYC
        df.drop(df[df.longitude < -74.28].index, inplace = True)
        df.drop(df[df.longitude > -73.65].index, inplace = True)
        df.drop(df[df.latitude < 40.48].index, inplace = True)
        df.drop(df[df.latitude > 40.93].index, inplace = True)
        
        shape_after = df.shape
        
        print(shape_after == df.shape)
        return df
    
    def crash_cleaner(self, df):
        
        self.df = df
        df = df.rename(columns = str.lower)
        df.columns = df.columns.str.replace(' ', '_')
        
        del df["off_street_name"] # 77% of values are NaN
        # removed to reduce necessary computation, since it is a useless information
        del df["on_street_name"]
        del df["cross_street_name"]
        del df["collision_id"]
        del df["zip_code"]
        
        
        df["accident_date"] = df["accident_date"].str.replace('T00:00:00.000', '')
        # To slice only accidents occured in 2017
        df.drop(df[df.accident_date >= "2018-01-01"].index, inplace = True)
        df.drop(df[df.accident_date <= "2016-12-31"].index, inplace = True)
        
        # To slice only accidents whose coordinates belong sto NYC 
        df.drop(df[df.longitude < -74.28].index, inplace = True)
        df.drop(df[df.longitude > -73.65].index, inplace = True)
        df.drop(df[df.latitude < 40.48].index, inplace = True)
        df.drop(df[df.latitude > 40.93].index, inplace = True)
        # reset index 
        df = df.reset_index(drop=True)
        # replacing NaNs with None
        df = df.where(pd.notnull(df),None)
        df['latitude'] = pd.to_numeric(df['latitude'], errors = 'raise')
        df['longitude'] = pd.to_numeric(df['longitude'], errors = 'raise')
        df['number_of_persons_injured'] = pd.to_numeric(df['number_of_persons_injured'], errors = 'raise')
        df['number_of_persons_killed'] = pd.to_numeric(df['number_of_persons_killed'], errors = 'raise')
        
        return df
    # --------------------------------------------------
    # Functions used to fill values in the column "borough" starting from coordinates
    # Haversine formula
    def distance(self, lat1, lon1, lat2, lon2):
        self.lat1 = lat1
        self.lat2 = lat2
        self.lon1 = lon1
        self.lon2 = lon2
        p = 0.017453292519943295 # math.PI / 180
        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
        return 12742 * asin(sqrt(a)) #2 * R; R = 6371 km

    # get the closest point 
    def closest(self, df, coords):
        self.df = df
        self.coords = coords
        return min(df, key=lambda p: self.distance(coords[0], coords[1], p[1], p[2]))
    
    
    # --------------------------------------------------