import pickle
import pandas as pd
import numpy as np

class Insurance( object ):
    def __init__( self ):
        self.home_path               = 'D:\Data Science\Comunidade DS\PA004'
        self.annual_premium_scaler   = pickle.load( open( self.home_path + '\parameters\\annual_premium_scaler.pkl', 'rb' ) )
        self.age_scaler              = pickle.load( open( self.home_path + '\parameters\\age_scaler.pkl', 'rb' ) )
        self.vintage_scaler          = pickle.load( open( self.home_path + '\parameters\\vintage_scaler.pkl', 'rb' ) )
        self.fe_region_code			 = pickle.load( open( self.home_path + '\parameters\\fe_region_code.pkl', 'rb' ) )
        self.fe_policy_sales_channel = pickle.load( open( self.home_path + '\parameters\\fe_policy_sales_channel.pkl', 'rb' ) )
       
    def data_preparation( self, df5 ):

        df3 = df5.copy()

        #column renaming

        #cols test data
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 
                    'vehicle_age', 'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage']

        #cols validation data
        #cols_new = ['id', 'gender', 'age', 'region_code', 'policy_sales_channel', 'driving_license', 
        #            'vehicle_age', 'vehicle_damage', 'previously_insured', 'annual_premium', 'vintage', 'response']

        df3.columns = cols_new

        #standardization
        #annual premium
        df3['annual_premium'] = self.annual_premium_scaler.fit_transform( df3[['annual_premium']].values )
        
        
        #reescaling
        #age
        df3['age'] = self.age_scaler.fit_transform( df3[['age']].values )
        
        #vintage
        df3['vintage'] = self.vintage_scaler.fit_transform( df3[['vintage']].values )
        
        
        #encoding
        #region_code - frequency encoding
        df3.loc[:, 'region_code'] = df3['region_code'].map( self.fe_region_code)

        #policy_sales_channel - frequency encoding
        df3.loc[:, 'policy_sales_channel'] = df3['policy_sales_channel'].map( self.fe_policy_sales_channel )

        #gender
        df3['gender'] = df3['gender'].apply( lambda x: 1 if x == 'Male' else 0)

        #vehicle_damage
        df3['vehicle_damage'] = df3['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)

        df3 = df3.fillna(0)

        cols_selected_importance = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured']
        
        return df3[ cols_selected_importance ]
        

    
    def get_prediction( self, model, original_data, test_data):
        
        #prediction
        pred = model.predict_proba( test_data )
        
        #merging prediction into original dataset
        original_data['score'] = pred[:,1].tolist()
        original_data = original_data.sort_values(by = 'score', ascending = False)
        #original_data = original_data.reset_index()
        
        return original_data.to_json( orient = 'records', date_format = 'iso')
        