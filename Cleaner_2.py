import statistics
import pandas as pd
import numpy as np
import scipy
from math import cos, asin, sqrt
from scipy import stats

class reshaper:
    
    def PVT_crash(self, df):
        self.df = df
        wdf = df.copy()

        pivot = pd.pivot_table(wdf,values = ["number_of_persons_injured", "number_of_persons_killed",
                                             "number_of_pedestrians_injured", "number_of_pedestrians_killed",
                                             "number_of_cyclist_injured", "number_of_cyclist_killed",
                                             "number_of_motorist_injured", "number_of_motorist_killed"],
                               index = ['neighborhood'], aggfunc = np.mean )
        pivot["neighborhood"] = pivot.index
        pivot = pivot[[list(pivot)[-1]] + list(pivot)[:-1]] # put "neighborhood" as first column
        pivot = pivot.reset_index(drop = True)
        return pivot
    
    def PVT_bnb(self, df):
        self.df = df
        wdf = df.copy()    
        pivot = pd.pivot_table(wdf, index = ['neighborhood'], values = ["room_type", "price", "minimum_nights", "number_of_reviews",
                                                                        "calculated_host_listings_count", "availability_365"],
                               aggfunc = {"room_type":stats.mode, "price":np.mean, "minimum_nights":np.mean,
                                          "number_of_reviews":np.mean, "calculated_host_listings_count":np.mean,
                                          "availability_365":np.mean})

        pivot["room_type"] = [ list(res[0])[0] for res in pivot["room_type"].tolist()]
        pivot["neighborhood"] = pivot.index
        pivot = pivot[[list(pivot)[-1]] + list(pivot)[:-1]]
        pivot = pivot.reset_index(drop = True)
        return pivot
    
    def merger(self, df_big, df2):
        self.df_big = df_big
        self.df2 = df2
        """
        Starting from the list of neighborhoods, the function decomposes the first parameter in a
        list of smaller dataframes. Then every sub-dataset is added with the right combination (row)
        of data in the if-else block; it is finally recomposed by a concatenation on the column axis
        """
        ne = list(dict.fromkeys(df_big["neighborhood"]))
        sublis = [ df_big[df_big.neighborhood == hood].reset_index(drop = True) for hood in ne ]
        max_len = max([len(x) for x in sublis])
    
        pvt_dict = dict(zip(list(df2["neighborhood"]), [list(df2.loc[x])[1:] for x in range(len(df2))]  ))                   
        nan_df = pd.DataFrame([[np.nan for x in range(df2.shape[1] - 1)] for x in range(max_len)])
    
        lis_mer = []
        for data in sublis:
            if data.iloc[0][1] in list(pvt_dict.keys()):
                row_m = pd.DataFrame([pvt_dict[data.iloc[0][1]] for x in range(len(data))])
                lis_mer += [pd.concat([data, row_m], axis = 1)]
            else:
                lis_mer += [pd.concat([data, nan_df[:len(data)]], axis = 1)]
            
        merge = pd.concat(lis_mer)
        cols = dict(zip( list(merge.columns)[- df2.shape[1] + 1:], list(df2.columns[1:]) ))
        merge = merge.rename(columns = cols)
        merge = merge.reset_index(drop = True)

        return merge