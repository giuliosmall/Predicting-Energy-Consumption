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
        
        
    def plot_2(self, df, m1, m2, m3):
        self.df = df
        
        size = int(len(df)*0.75)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows = 3, ncols = 2, figsize = (35, 15))
        space = np.linspace(0, 100)
        
        ax1.plot(space, m1.coef_[0][4]*space + m1.intercept_); ax1.set_title('Laundry °C')
        ax1.scatter(df.laundry_celsius[size:], df.appliance_wh[size:])

        ax2.plot(space, m1.coef_[0][1]*space + m1.intercept_); ax2.set_title('Kitchen Humidity %')
        ax2.scatter(df.kitchen_hum_perc[size:], df.appliance_wh[size:])

        ax3.plot(space, m2.coef_[0][3]*space + m2.intercept_); ax3.set_title('Laundry °C LASSO')
        ax3.scatter(df.laundry_celsius[size:], df.appliance_wh[size:])

        ax4.plot(space, m2.coef_[0][1]*space + m2.intercept_); ax4.set_title('Kitchen Humidity % LASSO')
        ax4.scatter(df.kitchen_hum_perc[size:], df.appliance_wh[size:])

        ax5.plot(space, m3.coef_[0][4]*space + m3.intercept_); ax5.set_title('Laundry °C RFECV')
        ax5.scatter(df.laundry_celsius[size:], df.appliance_wh[size:])

        ax6.plot(space, m3.coef_[0][1]*space + m3.intercept_); ax6.set_title('Kitchen Humidity % RFECV')
        ax6.scatter(df.kitchen_hum_perc[size:], df.appliance_wh[size:])
        plt.savefig("linear_model.jpg", dpi = 300)