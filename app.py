

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

homes = pd.read_csv('WestRoxbury.csv')

homes = homes.rename( columns={
    'TOTAL_VALUE ' : 'TOTAL_VALUE',
    'LOT_SQFT '    : 'LOT_SQFT',
    'FLOORS '      : 'FLOORS',
    'BEDROOMS '    : 'BEDROOMS',
    'GROSS_AREA'   : 'GROSS_AREA',
} )

homes['REMODEL'] = homes['REMODEL'].replace('None', 0).replace('Recent', 1).replace('Old', 2)
homes.head()

from sklearn.linear_model import LinearRegression

predictors = homes.iloc[:,2:]
response = homes.iloc[:, 0]

model = LinearRegression()
model.fit( predictors, response )

predictions = model.predict( predictors )
actual = homes['TOTAL_VALUE']
mse = ( ( predictions - actual ) ** 2 ).mean()
rmse = mse ** .5
#rmse

#len( homes )

import numpy as np
rows_for_training = np.random.choice( homes.index, 3842, False )
training = homes.index.isin( rows_for_training )
df_training = homes[training]
df_validation = homes[~training]
#len( df_training ), len( df_validation )

# model based on training data
train_predictors = df_training.iloc[:,2:]
train_response = df_training.iloc[:,0]
model= LinearRegression()
model.fit( train_predictors, train_response )

# same code as earlier, just on a different dataframe
train_predictions = model.predict( df_training.iloc[:,2:] )
train_actual = df_training['TOTAL_VALUE']
train_mse = ( ( train_predictions - train_actual ) ** 2 ).mean()
train_rmse = train_mse ** .5

# same code again, now on the validation data
validation_predictions = model.predict( df_validation.iloc[:,2:] )
validation_actual = df_validation['TOTAL_VALUE']
validation_mse = ( ( validation_predictions - validation_actual ) ** 2 ).mean()
validation_rmse = validation_mse ** .5

#train_rmse, validation_rmse

def fit_model_to ( training ):
    predictors = training.iloc[:,2:]
    response = training.iloc[:,0]
    model = LinearRegression()
    model.fit( predictors, response )
    return model

def score_model ( M, validation ):
    predictions = M.predict( validation.iloc[:,2:] )
    actual = validation['TOTAL_VALUE']
    mse = ( ( predictions - actual ) ** 2 ).mean()
    rmse = mse ** .5
    return rmse

model = fit_model_to( df_training )
#score_model( model, df_training ), score_model( model, df_validation )

def fit_model_to ( training ):
    # choose predictors and fit model as before
    predictors = training.iloc[:,2:]
    response = training.iloc[:,0]
    model = LinearRegression()
    model.fit( predictors, response )
    # fit another model to standardized predictors
    standardized = ( predictors - predictors.mean() ) / predictors.std()
    temp_model = LinearRegression()
    temp_model.fit( standardized, response )
    # get that model's coefficients and display them
    coeffs = pd.Series( temp_model.coef_, index=predictors.columns )
    sorted = np.abs( coeffs ).sort_values( ascending=False )  # these two lines are the
    coeffs = coeffs.loc[sorted.index]                         # optional bonus, sorting
    #print( coeffs )
    # return the model fit to the actual predictors
    return model

# make sure it works
model = fit_model_to( df_training )
#print( score_model( model, df_training ), score_model( model, df_validation ) )

columns = [0,1,2,3,4,5,6,9,10,11,12,13]
model = fit_model_to( df_training.iloc[:,columns] )
#score_model( model, df_training.iloc[:,columns] ), score_model( model, df_validation.iloc[:,columns] )

st.title("Predict the Value of a Home in West Roxbury, MA")

living_area = st.number_input('Enter Living Area:',0,6000,0, step=100)
gross_area = st.number_input('Enter Gross Area:',0,10000,0,step=100)
lot_sqft = st.number_input('Enter Lot Square Feet:',0,15000,0, step=100)
yr_built = st.number_input('Enter Year Built:',1700,2021,1970)
floors = st.number_input('Enter Number of Floors:',1,10,1)
full_bath = st.number_input('Enter Number of Full Baths:',0,10,0)
remodel1 = st.selectbox('Type of Remodel:', ('Recent', 'Old', 'None'))
half_bath = st.number_input('Enter Number of Half Baths:',0,10,0)
kitchen= st.number_input('Enter Number of Kitchens:',0,10,0)
fireplace = st.number_input('Enter Number of Fireplaces:',0,10,0)

if remodel1 == 'Recent':
    remodel = 1
elif remodel1 == 'Old':
    remodel = 2
else:
    remodel = 0


column_names = ['TOTAL_VALUE',
                'TAX',
                'LOT_SQFT',
                'YR_BUILT',
                'GROSS_AREA',
                'LIVING_AREA',
                'FLOORS',
                'ROOMS',
                'BEDROOMS',
                'FULL_BATH',
                'HALF_BATH',
                'KITCHEN',
                'FIREPLACE',
                'REMODEL']
data = [0, 0, lot_sqft, yr_built, gross_area, living_area, floors, 0, 0, full_bath, half_bath, kitchen, fireplace, remodel]

df_dashboard = pd.DataFrame([data], columns=column_names)

columns = [0,1,2,3,4,5,6,9,10,11,12,13]
predicted_home_value = score_model( model, df_dashboard.iloc[:,columns] ) #score_model works here because "actual" home value in the new dataframe is 0
real_home_value = predicted_home_value * 1000
st.write(f'Based on your inputs, the predicted value of the home in West Roxbury, MA will be approximately ${real_home_value:,.2f}.')

