![cooltext424750920217702](https://user-images.githubusercontent.com/15370068/205403747-eab3181d-9297-499e-9ee8-90a993d4732e.png)

## Objective
This project aims to find the environmental variables that best explain solar power output, in the absence of irradiation data.

### Methods Used
- Data Visualization
- Correlation Plots
- Regression Models (from sklearn):
	- LinearRegression
	- MLPRegressor
	- RandomForestRegressor
	- HistGradientBoostingRegressor
	- GradientBoostingRegressor
	- AdaBoostRegressor
	- StackingRegressor
	- DecisionTreeRegressor

### Technologies
- Python
- Numpy, Pandas
- Matplotlib, Seaborn, Plotly
- Sklearn

### Data
We use the following dataset: 
- Williams, Jada; Wagner, Torrey (2019), “Northern Hemisphere Horizontal Photovoltaic Power Output Data for 12 Sites”, Mendeley Data, V5, doi: 10.17632/hfhwmn8w24.5 

This dataset contains power output from horizontal photovoltaic panels located at 12 Northern hemisphere sites over 14 months. It has 21K observations of 17 variables as follows: location, date, time sampled, latitude, longitude, altitude, year and month, month, hour, season, humidity, ambient temperature, power output from the solar panel, wind speed, visibility, pressure, and cloud ceiling. 

## Project Description
Solar energy is a rapidly growing market. In order to maintain solar plant sites and produce the required amount of energy, knowing solar power output is vital. This is normally modeled with irradiance data. But irradiance measurements require specific sensors, which take a long time to deploy and have a high cost. Hence, finding a way to predict solar power output with existing weather and location data would certainly be beneficial. We use Exploratory Data Analysis (EDA) along with a variety of regression models to dissect the variables that best explain solar power output. We conclude that these variables are: climate type, ambient temperature, humidity, and cloud ceiling.

## Repository Contents
- `Pasion et al dataset.csv`: data 
- `eda_notebook.ipynb`: jupyter notebook containing all EDA; can also be viewed with Google Colab [here](https://colab.research.google.com/drive/1lwJoR0XxA76lOJT2g6u6Y5HqkZaizeZT?usp=sharing)
- `kde_plot.py`, `model_testing.py`: external functions used for EDA and predictions
- `presentation.pdf`: PDF of our presentation
