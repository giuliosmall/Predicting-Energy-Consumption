import statistics
import pandas as pd
import numpy as np
import scipy


class cleaners:
    
    def house_price_cleaner(self, df):
        
        self.df = df
        
        # Drop EASE-MENT column since it is empty, and UNNAMED: 0
        del df["EASE-MENT"]
        del df["Unnamed: 0"]
        
        # Change columns' names in order to have lower case and no spasce between words
        df = df.rename(columns=str.lower)
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
        df["sale_price"] = df["sale_price"].replace(' -  ', "%%%%")

        ######## ARE THOSE LINES TO BE DROPPED OR TO ADD THE MEAN
        df.drop(df[df.sale_price == "%%%%"].index, inplace = True)

        # Convert all the remaining values to float
        df['sale_price']= pd.to_numeric(df['sale_price'], errors = 'raise')

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
        
        return df