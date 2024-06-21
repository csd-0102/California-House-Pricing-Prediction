# California-House-Pricing-Prediction
In this project, we aimed to predict house sale prices in California using a dataset that includes various features about the houses such as the number of bedrooms, living areas, locations, nearby schools, and seller summaries. Below are the detailed steps taken to complete the project:

1. Data Loading
    We began by loading the training and testing datasets from Google Drive.
    The datasets were in CSV format and contained house-related information for houses sold in California in 2020.
2. Data Exploration and Visualization
    We explored the training data to understand its structure and contents using pandas and matplotlib.
    Histograms were plotted to visualize the distribution of numerical features.
    A correlation matrix was created to see how different features correlate with each other.
3. Data Preprocessing
    Handling Dates: Converted the 'Listed On' date to a datetime format and extracted the year.
    Feature Engineering: Created new features such as:
    age: Difference between the listing year and the year the house was built.
    Total School Distance: Sum of distances to elementary, middle, and high schools.
    Total School Score: Sum of scores for elementary, middle, and high schools.
    Total Area: Sum of lot area, livable area, and total spaces.
    Bathroom Ratio: Ratio of full bathrooms to total bathrooms.
    Handling Missing Values: Used SimpleImputer to fill missing values in numerical columns with the mean.
    Log Transformation: Applied log transformation to skewed features to make their distributions more normal.
4. Feature Scaling
    Applied StandardScaler to standardize the numerical features.
    Scaled the target variable (house prices) to normalize it.
5. Model Selection and Training
    Linear Regression: Trained a linear regression model as a baseline.
    Decision Tree Regression: Used a decision tree regressor to capture non-linear relationships.
    Random Forest Regression: Implemented a random forest regressor which uses multiple decision trees for better prediction accuracy.
    Hyperparameter Tuning: Used RandomizedSearchCV to find the best hyperparameters for the RandomForestRegressor.
6. Model Evaluation
    Evaluated models using Root Mean Squared Error (RMSE) to measure the prediction accuracy.
    Performed cross-validation to ensure the model's performance is consistent across different subsets of the data.
7. Model Prediction
    Used the best model (RandomForestRegressor with tuned hyperparameters) to predict house prices on the test dataset.
    Saved the predictions to a text file named predictions_housing.txt.
