'''
MODEL CONFIDENCE LEVEL 

- NORMAL DAYS 
- PEAK DAYS 

@author: Arooj Ahmed Qureshi
@Contributors: Habib , Eugene 
@Property of: Enpowered Inc 
'''
import pandas as pd
import datetime as datetime
import numpy as np

class model_confidence:
	'''
	Returns the model confidence level percentage during `Normal Days` and during `Peak Days`
	
	Parameters
	-----------
	path : Folder location where .csv file containing model prediction and actual forecast. (it should be 1 csv file). 
	region: [String] For Example: ERCOT, AESO, NYISO etc. 
	threshold: [number] value set for the specific region

	'''
	def __init__(self, path, region, threshold):
		self.region = region
		self.path = path
		self.peak_threshold = threshold

	def __call__(self):
		print(f"MODEL : BASE - ENSEMBLE {self.region}")
		# Load the model prediction and actual load `csv` file in pandas dataframe. 
		self.dataframe = pd.read_csv(self.path)

		# rename column names. actual: represents original load. 'value': represents model predicted value.
		self.dataframe = self.dataframe.rename({'local_datetime': 'Date_time','live':'actual'}, axis=1)

		# Separate base and ensemble model forecast and original 
		# BASE MODEL
		model_base_forecast = self.dataframe[['Date_time', 'value', 'actual']]
		model_base_forecast.index = pd.to_datetime(model_base_forecast['Date_time']) 
		model_base_forecast.Date_time = pd.to_datetime(model_base_forecast['Date_time']) 
		model_base_forecast['Date'] = model_base_forecast.Date_time.dt.date
		#print(f'BASE MODEL \n {model_base_forecast.head()}')

		# ENSEMBLE MODEL
		model_ensemble_forecast = self.dataframe[['Date_time', 'ensemblevalue','actual']]
		model_ensemble_forecast.index = pd.to_datetime(model_ensemble_forecast['Date_time'])
		model_ensemble_forecast.Date_time = pd.to_datetime(model_ensemble_forecast['Date_time']) 
		model_ensemble_forecast['Date'] = model_ensemble_forecast.Date_time.dt.date
		model_ensemble_forecast = model_ensemble_forecast.rename({'ensemblevalue': 'value'}, axis=1)
		#print(f'ENSEMBLE MODEL \n {model_ensemble_forecast.head()}')
		
		# EXTRACT TOP N PEAK DAYS 
		self.peak_data = model_base_forecast.groupby(['Date']).max()
		
		number_of_days = 10
		top_peak_days = self.peak_data.nlargest(number_of_days, 'actual')
		self.peak_data = self.peak_data.loc[top_peak_days.index]
		self.peak_date = top_peak_days
		self.peak_data.to_csv("ERCOT_BASE_PEAK_ACTUAL.csv")
		print("ACTUAL PEAK DAYS")
		print(top_peak_days)


		# BASE MODEL TOP N PEAK DAYS
		print("BASE MODEL PEAK DAYS")
		print(self.peak_data.nlargest(number_of_days, 'value'))
		model_peak_info = self.peak_data.nlargest(number_of_days, 'value')
		model_peak_info.to_csv("ERCOT_BASE_PEAK_VALUE.csv")

		# META MODEL TOP N PEAK DAYS
		#print("META MODEL PEAK DAYS")
		self.meta_peak_data = model_ensemble_forecast.groupby(['Date']).max()
		top_meta_peak_days = self.meta_peak_data.nlargest(number_of_days, 'value')
		self.meta_peak_data = self.meta_peak_data.loc[top_meta_peak_days.index]
		#print(self.meta_peak_data.nlargest(number_of_days, 'value'))

		# MATRIX-1: MODEL PERFORMANCE ON THE PEAK DAYS 
		# MATRIX-2: MODEL PEROFRMANCE ON THE PEAK DAYS w.r.t. THE SET THRESHOLD 
		day_result_base = self.peak_day_mae('BASE')
		day_result_ensemble = self.peak_day_mae('ENSEMBLE')

		# MATRIX-3 MODEL PERFORMANCE ON THE PEAK DAYS w.r.t. EXACT PEAK HOUR WINDOW
		hourly_result_base= self.peak_hour_accuracy(model_base_forecast, 'BASE')
		hourly_result_ensemble= self.peak_hour_accuracy(model_ensemble_forecast, 'ENSEMBLE')

		return None#day_result, hourly_result

	def thres_mae_accuracy(self, max_value, max_actual):
		if(max_actual>self.peak_threshold):
			#print('PEAK DAYS ANALYSIS')
			# PEAK DAYS ANALYSIS
			if(max_value>self.peak_threshold):
				return 0.9
			elif((self.peak_threshold - max_value)> 500):
				return 0.2
			elif((self.peak_threshold - max_value)< 500):
				return 0.7
			else:
				return 0.7
		elif(max_actual<self.peak_threshold):
			# REGULAR DAYS ANALYSIS
			#print('REGULAR DAYS ANALYSIS')
			if((self.peak_threshold - max_actual)<500):
				if((self.peak_threshold - max_value)> 500):
					return 0.2
				elif((self.peak_threshold - max_value)< 500):
					return 0.9
				else:
					return 0.7
			else:
				if((abs(max_actual - max_value)<500)):
					return 0.9
				elif((abs(max_actual - max_value)>500) and (abs(max_actual - max_value)<1000)):
					return 0.6
				else:
					return 0.2
		else:
			return 0.1

	def peak_day_mae(self, title):
		if title == 'BASE':
			print(f'PEAK DAY MAE {title}')
			model_peak_data = self.peak_data
		elif title == 'ENSEMBLE':
			print(f'PEAK DAY MAE {title}')
			model_peak_data = self.meta_peak_data
		else:
			print(f'{title} model does not exist.')

		max_actual_load = model_peak_data['actual'].max()
		model_peak_data['Date'] = model_peak_data.index
		model_peak_data.index = range(len(model_peak_data))
		df_er = pd.DataFrame(columns=['Date', 'MAE', 'Thres_MAE_Confidence'], index=range(len(self.peak_date)) )
		for i in range(len(model_peak_data)):
			MAE = np.abs(model_peak_data.loc[i,'actual'] - model_peak_data.loc[i,'value'])
			date = model_peak_data.loc[i,'Date']
			max_forecast = model_peak_data.loc[i,'value']
			mae_accuracy = self.thres_mae_accuracy(model_peak_data.loc[i,'value'],model_peak_data.loc[i,'actual'])

			#print(f'Date: {date}, mae_accuracy {mae_accuracy}')
			thres_level = ((1 -mae_accuracy ) * MAE )

			df_er.loc[i,'Date'] = date
			df_er.loc[i,'MAE'] = MAE
			df_er.loc[i,'Thres_MAE_Confidence'] = thres_level

		
		df_er = df_er.dropna() 
		
		print('MAE per day \n', df_er)
		mean_mae = df_er['MAE'].median()
		mean_thres_mae = df_er['Thres_MAE_Confidence'].median()
		print(f'max_actual_load: {max_actual_load}')
		print(f'Mean absolute of (max_forecast - max_load) of peak days {mean_mae} and error % = {(mean_mae/max_actual_load)*100}')
		print(f'Mean absolute of thres % *(max_forecast - max_load) of peak days error% {(mean_thres_mae/max_actual_load)*100} and confidence % = {(1 - (mean_thres_mae/max_actual_load))*100}')

	def peak_hour_accuracy(self, model, title):   

		#peak_date = self.peak_date
		#df_2021 = self.df_final

		df_er = pd.DataFrame(columns=['date', '1st_forecast_p_hour_align?','2nd_forecast_p_hour_align?', '3rd_forecast_p_hour_align?' ], index=range(0,len(self.peak_date)) )    
		for i, date in enumerate(self.peak_date.index):
			df_temp = model[model.index.date == date ]

			if (df_temp['actual'].empty==True):
				continue

			df_er.iloc[i,0] = date

			peak_h = df_temp['actual'].idxmax()
				
			forecast_peak_h_1 = df_temp['value'].nlargest(3).index[0]
			forecast_peak_h_2 = df_temp['value'].nlargest(3).index[1]
			forecast_peak_h_3 = df_temp['value'].nlargest(3).index[2]
			
			df_er.iloc[i:,1] = [forecast_peak_h_1 == peak_h]
			df_er.iloc[i:,2] = [forecast_peak_h_2 == peak_h]
			df_er.iloc[i:,3] = [forecast_peak_h_3 == peak_h]
			
		df_er = df_er.dropna()     
		df_er.iloc[:, 1:4] = df_er.iloc[:, 1:4].astype(int)
		acc_1_peak_hour = (df_er.iloc[:, 1].sum()/df_er.iloc[:,1].count())*100
		print(f'HOUR ACCURACY FOR THE MODEL: {str(title)}')
		print("Accuracy of top 1 forecast hour" ,acc_1_peak_hour, '%')       


		acc_2_peak_hour = (df_er.iloc[:, 2].sum()/df_er.iloc[:,2].count())*100
		print("Accuracy of top 2 forecast hour" ,acc_2_peak_hour, '%') 

		acc_3_peak_hour = (df_er.iloc[:, 3].sum()/df_er.iloc[:,3].count())*100
		print("Accuracy of top 3 forecast hour" ,acc_3_peak_hour, '%') 

		acc_1_or_2_or_3_peak_hour = (df_er.iloc[:, 1:4].sum().sum()/len(df_er.iloc[:,:]))*100
		print("Accuracy of any of the top 3 forecast aligns with actual" ,acc_1_or_2_or_3_peak_hour, '%')

		return df_er

def main():
	path = '../Data/NYISO_September_combine_data.csv'
	#path = '../Data/NYISO_August_combine_data.csv'
	peak_day = model_confidence(path, 'NYISO', 29000 )

	#path = '../Data/ERCOT_September_combine_data.csv'
	#peak_day = model_confidence(path, 'ERCOT', 70000)
	peak_day()
	
if __name__ == '__main__':
	main()
