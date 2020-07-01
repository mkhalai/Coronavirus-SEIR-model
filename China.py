import numpy as np
import pandas as pd

cases = pd.read_csv('cases.csv')
recov = pd.read_csv('recovered.csv')
deaths = pd.read_csv('deaths.csv')

class China:
    """
    All data sourced from CSSE John Hopkins University
    (https://github.com/CSSEGISandData)
    """
    def __init__(self):
        self.china_cases = cases[cases['Country/Region']=='China']
        self.china_recov = recov[recov['Country/Region']=='China']
        self.china_deaths = deaths[deaths['Country/Region']=='China']

        #Drop useless columns

        self.china_cases.drop(columns = ['Lat','Long','Country/Region'],inplace=True)
        self.china_recov.drop(columns = ['Lat','Long','Country/Region'],inplace=True)
        self.china_deaths.drop(columns = ['Lat','Long','Country/Region'],inplace=True)

        g = pd.concat([self.china_recov,self.china_deaths]).groupby('Province/State')
        self.removed = g.agg('sum').reset_index()

    def get_cases_removed(self,province):
        """
        Return cases and removed for a given province
        """
        province = province.capitalize()
        prov_cases = self.china_cases[self.china_cases['Province/State']==province].iloc[:,1:]
        prov_rem = self.removed[self.removed['Province/State']==province].iloc[:,1:]
        join = pd.concat([prov_cases,prov_rem]).T
        join.columns = ['cases','removed']

        return join

    def get_provinces(self):
        return self.china_cases['Province/State'].values
