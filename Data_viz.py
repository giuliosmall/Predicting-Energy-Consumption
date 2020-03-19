import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

class Graphs:
    
    def plot_1(self, df):
        self.df = df
    
        # Set up the plotting layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (35, 15))
        fig.autofmt_xdate(rotation = 45)

        # TEMPERATURE

        # Living Room 
        ax1.plot(df.date, df.living_celsius, linewidth = 1, alpha = 0.7, label = "Living Room")
        ax1.set_xlabel('Nov 1st 2016 - May 27th 2016'); ax1.set_ylabel('Temp in Celsius'); ax1.set_title('Temperature Trend')
        # Parents Room 
        ax1.plot(df.date, df.parents_celsius, linewidth = 1, alpha = 0.7, label = "Parents Room")
        ax1.set_xlabel('Nov 1st 2016 - May 27th 2016');
        # Ironing Room 
        ax1.plot(df.date, df.ironing_celsius, linewidth = 1, alpha = 0.7, label = "Ironing Room")
        ax1.set_xlabel('Nov 1st 2016 - May 27th 2016'); 
        ax1.legend()
    
        # HUMIDITY Outside vs HUMIDITY Wheather Station
    
        # Outiside
        ax2.plot(df.date, df.portico_hum_perc, linewidth = 1, alpha = 0.7, label = "North Outside House")
        ax2.set_xlabel('Nov 1st 2016 - May 27th 2016'); ax2.set_ylabel('Humidity in %'); ax2.set_title('Humidity Trend')
        # Weather Station
        ax2.plot(df.date, df.cws_hum_perc, linewidth = 1, alpha = 0.7, label = "Weather Station")
        ax2.set_xlabel('Nov 1st 2016 - May 27th 2016')
        ax2.legend()
    
        # OUTSIDE HUMIDITY vs VISIBILITY 
    
        # Outside Humidity
        ax3.plot(df.date, df.cws_hum_perc, linewidth = 1, alpha = 0.7, label = "Outside Humidity")
        ax3.set_xlabel('Nov 1st 2016 - May 27th 2016'), ax3.set_title('Humidity vs Visibility')
        # Visibility
        ax3.plot(df.date, df.cws_visibility, linewidth = 1, alpha = 0.7, label = "Visibility")
        ax3.set_xlabel('Nov 1st 2016 - May 27th 2016')
        ax3.legend()
    
        # TEMPERATURE Outside vs TEMPERATURE Wheather Station
        # Outiside
        ax4.plot(df.date, df.portico_celsius, linewidth = 1, alpha = 0.7, label = "North Outside House")
        ax4.set_xlabel('Nov 1st 2016 - May 27th 2016'); ax2.set_ylabel('Temp in Celsius'); ax4.set_title('Temperature Trend')
        # Weather Station
        ax4.plot(df.date, df.cws_celsius, linewidth = 1, alpha = 0.7, label = "Weather Station")
        ax4.set_xlabel('Nov 1st 2016 - May 27th 2016')
        ax4.legend()

        plt.tight_layout(pad=2)
        plt.savefig("Relations.jpg", pdi = 300)