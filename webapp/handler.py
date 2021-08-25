import os
import pickle

import pandas as pd

from flask import Flask, request, Response

from insurance.Insurance import Insurance

#loading model
model = pickle.load( open( 'model/xgbclassifier.pkl', 'rb' ) )


#API intialization
app = Flask( __name__ )

@app.route( '/insurance/predict', methods = ['POST'] )

def insurance_predict():
	test_json = request.get_json()
    
	if test_json: # there is data
		if isinstance( test_json, dict ): #single example
			test_raw = pd.DataFrame( test_json, index = [0] )
            
		else:  #multiple example      
			test_raw = pd.DataFrame( test_json, columns = test_json[0].keys() )
            
		# instantiate insurance class
		pipeline = Insurance()
        
		#data preparation
		df3 = pipeline.data_preparation( test_raw )
        
		#prediction
		df_response = pipeline.get_prediction( model, test_raw, df3 )
        
		return df_response
            
	else: # no data
		return Response( '{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
	port = os.environ.get( 'PORT', 5000)
	app.run(host = '0.0.0.0', port = port )