import pandas as pd

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, \
    AdaBoostRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
A series of model performance testing.

The final optimal model is stacking model, which is
essentially a mixture(combination) of models which have best performance
in different climate type.

The initial reason to use stacking model is based on a simple observation:
    GLM performs better in hot-dry, cold-humid, cold-dry climates
    GradientBoost, HistogramGradientBoost perform better in hot-humid climate
    
So a "stacked" model from GLM, GradientBoost, and HistogramGradientBoost is constructed

For more info, please go to:
def StackingRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df)

Zhenduo Wen
"""


def evaluationMetric(Y_test, Y_pred):
    """
    :param Y_test: test dataset
    :param Y_pred: true value
    :return: list, contains metrics
    """

    # evaluation metrics to use
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    return [rmse, mae, r2]


def linearRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Simple Linear Regression
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # fit model
    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['Linear', climate_name, len(X_train.index), *metrics]


def bayesianRidgeTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Bayesian Ridge Regression
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """
    # fit model
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['BayesianRidge', climate_name, len(X_train.index), *metrics]


def GLMTest(X_train, Y_train, X_test, test, climate_name, output_option, output_df,
            degree=3, show_sample_prediction=False, show_outliers=False):
    """
    Generalized Linear Model
    https://scikit-learn.org/0.15/modules/linear_model.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # fit model
    Y_test = test['polypwr']
    reg = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression(fit_intercept=True))])
    reg = reg.fit(X_train, Y_train)

    Y_pred = reg.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['GLM', climate_name, len(X_train.index), *metrics]

    if show_sample_prediction:
        test_predict = pd.Series(reg.predict(X_test), index=Y_test.index)
        test_df = pd.concat([test_predict, Y_test], axis=1)

        test_df.set_axis(['pred', 'real'], axis=1, inplace=True)
        print("prediction sample: ")
        print(test_df.head())

    if show_outliers:

        test_predict = pd.Series(test_predict, index=Y_test.index)
        test_df = pd.concat([test_predict, Y_test], axis=1)

        test_df.set_axis(['pred', 'real'], axis=1, inplace=True)

        C = 3
        df_outliers = pd.DataFrame()
        rmse = mean_squared_error(Y_test, Y_pred, squared=False)

        for index, row in test_df.iterrows():
            if abs(row.pred - row.real) > C * rmse:
                df_outliers = df_outliers.append(test.loc[index])
            else:
                test_df.drop(index, inplace=True)

        print(df_outliers)
        print(test_df)


def AdaBoostRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    AdaBoosting Regression
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """
    # fit model
    regr = AdaBoostRegressor(random_state=42, n_estimators=100)
    regr.fit(X_train, Y_train)

    Y_pred = regr.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['AdaBoostReg', climate_name, len(X_train.index), *metrics]


def HistGradBoostRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Histogram Gradient Regression
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """
    # fit model
    regr = HistGradientBoostingRegressor(random_state=42).fit(X_train, Y_train)

    Y_pred = regr.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['HistGradBoost', climate_name, len(X_train.index), *metrics]


def DecisionTreeRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Decision Tree Regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # fit model
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, Y_train)

    Y_pred = reg.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['DecisionTree', climate_name, len(X_train.index), *metrics]


def RFRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Random Forest Regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # fit model
    regr = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
    regr.fit(X_train, Y_train)

    Y_pred = regr.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['RandomForest', climate_name, len(X_train.index), *metrics]


def GradBoostRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Gradient Boosting Regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # fit model
    reg = GradientBoostingRegressor(random_state=0, max_depth=5)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['GradientBoosting', climate_name, len(X_train.index), *metrics]


def MLPNNRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Multiple Layer Neural Network
    default number of layers be 100
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # fit model
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['MultiLayerNN', climate_name, len(X_train.index), *metrics]


def KNNRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df, n_neighbors, weights):
    """
    KNN Regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :param n_neighbors: int, number of neighbors
    :param weights: weights for distance
    :return: None
    """
    assert isinstance(n_neighbors, int)
    assert n_neighbors > 0

    # fit model
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    Y_pred = knn.fit(X_train, Y_train).predict(X_test)

    # store test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['KNNReg', climate_name, len(X_train.index), *metrics]


def StackingRegTest(X_train, Y_train, X_test, Y_test, climate_name, output_option, output_df):
    """
    Stacked model with structure:

    **************************************************************************
    General Linear Model --------
                                  \
    Gradient Boosting Model -----------> General Linear Model --> prediction
                                 /
    Histogram Boosting Model ---
    **************************************************************************

    combining the three models that have the best performance in different climates

    In general, Stacking Model achieves the best overall performance, but the price is
    much more computational resource.

    :param X_train: pd.DataFrame, training feature dataset
    :param Y_train: pd.DataFrame, training result dataset
    :param X_test: pd.DataFrame, testing feature dataset
    :param Y_test: pd.DataFrame, testing result dataset
    :param climate_name: string, climate type name
    :param output_option: default as "to_dataframe", store testing result to output dataframe
    :param output_df: pd.DataFrame, dataframe to hold testing results
    :return: None
    """

    # estimators for the stacked model
    estimators = [
        ('glm', Pipeline([('poly', PolynomialFeatures(degree=3)),
                          ('linear', LinearRegression(fit_intercept=True))])),
        ('gradboost', GradientBoostingRegressor(random_state=42, max_depth=5)),
        ('histboost', HistGradientBoostingRegressor(random_state=42))
    ]

    # construct stack model
    # use GLM as final estimator
    regr = StackingRegressor(
        estimators=estimators,
        final_estimator=Pipeline([('poly', PolynomialFeatures(degree=3)),
                                  ('linear', LinearRegression(fit_intercept=True))])
    )

    # model fitting
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)

    # output test result
    if output_option == "to_dataframe":
        metrics = evaluationMetric(Y_test, Y_pred)
        output_df.loc[len(output_df.index)] = ['StackingReg', climate_name, len(X_train.index), *metrics]


def Test(climate_name, climate_dict, df, output_df, output_option='to_dataframe'):
    """
    Test funtion.
    Implement tests for models
    :param climate_name: string, a key from climate_dict
    :param climate_dict: dictionary, key (climate_name): locations (location_name)
    :param df: pd dataframe to perform model fitting (normalized)
    :param output_df: pd dataframe to save test result
    :param output_option: default as 'to_dataframe'
    :return: None, add test result to output_df
    """
    assert isinstance(climate_name, str)
    assert isinstance(climate_dict, dict)

    TEST_SET_SIZE = 0.1

    # pick x features to load into the model
    x_cols = ['humidity', 'ambient_temp', 'wind_speed',
              'pressure', 'cloud_ceiling', 'latitude', 'longitude', 'altitude', 'month',
              'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']

    locations = climate_dict[climate_name]
    df = df[df['location'].isin(locations)]

    # Perform train-test split
    train, test = train_test_split(df, test_size=TEST_SET_SIZE, shuffle=True, random_state=42)

    X_train = train[x_cols]
    Y_train = train["polypwr"]
    X_test = test[x_cols]
    Y_test = test["polypwr"]

    # Do model testing

    # linear models
    linearRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                  output_df=output_df)
    bayesianRidgeTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                      output_df=output_df)
    GLMTest(X_train, Y_train, X_test, test, show_sample_prediction=False, show_outliers=False,
            climate_name=climate_name, output_option=output_option, output_df=output_df)

    # clustering regression models
    KNNRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
               output_df=output_df, n_neighbors=10, weights='uniform')

    # decision tree and random forests
    DecisionTreeRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                        output_df=output_df)
    RFRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
              output_df=output_df)

    # multiple layer neural network
    MLPNNRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                 output_df=output_df)

    # Boosting and Stacking
    AdaBoostRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                    output_df=output_df)
    GradBoostRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                     output_df=output_df)
    HistGradBoostRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                         output_df=output_df)
    StackingRegTest(X_train, Y_train, X_test, Y_test, climate_name=climate_name, output_option=output_option,
                    output_df=output_df)


# local testing
# reminder: local test estimated running time on a laptop: 1min30s
if __name__ == "__main__":

    # load dataframe and convert column names to lowercase
    df = pd.read_csv("Pasion et al dataset.csv")
    df.columns = df.columns.str.lower()
    dummy = pd.get_dummies(df.season, prefix='season')
    df_with_season_code = pd.concat([df, dummy], axis=1);
    df.columns = df.columns.str.replace('.', '_')
    df = df.rename(columns={'ambienttemp': 'ambient_temp'})

    # normalize columns
    numeric_cols = ['humidity', 'ambient_temp', 'wind_speed',
                    'pressure', 'cloud_ceiling', 'latitude', 'longitude', 'altitude', 'month']
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # one hot encoding
    dummy = pd.get_dummies(df.season, prefix='season')
    df_with_season_code = pd.concat([df, dummy], axis=1);

    # assign climate type to each location
    climate_dict = {'hot-dry': ['March AFB', 'Travis'],
                    'cold-dry': ['Hill Weber', 'USAFA', 'Peterson', 'Offutt'],
                    'cold-humid': ['Grissom', 'Malmstrom', 'MNANG', 'Camp Murray'],
                    'hot-humid': ['JDMT', 'Kahului'],
                    'mixture': df.location.unique()
                    }

    # initialize output dataframe
    output_df = pd.DataFrame(columns=['Model', 'Climate', 'TrainSetSize', 'MSE', 'MAE', 'R^2'])

    # do testing for each climate type
    for climate_name in climate_dict.keys():
        Test(climate_name, climate_dict, df_with_season_code, output_df=output_df)

    # show testing result
    print(output_df)

    """
    Sample Output:
    in which shows Stacking Model has the best overall performance
    
                   Model     Climate  TrainSetSize       MSE       MAE       R^2
    0             Linear     hot-dry          4455  3.459830  2.585673  0.667328
    1      BayesianRidge     hot-dry          4455  3.461362  2.586513  0.667033
    2                GLM     hot-dry          4455  3.024271  2.223471  0.745816
    3             KNNReg     hot-dry          4455  3.255319  2.326237  0.705494
    4       DecisionTree     hot-dry          4455  4.059189  2.794442  0.542084
    5       RandomForest     hot-dry          4455  3.371985  2.514834  0.684006
    6       MultiLayerNN     hot-dry          4455  3.167798  2.249535  0.721117
    7        AdaBoostReg     hot-dry          4455  4.262408  3.534271  0.495087
    8   GradientBoosting     hot-dry          4455  3.055259  2.210929  0.740580
    9      HistGradBoost     hot-dry          4455  3.113155  2.212018  0.730655
    10       StackingReg     hot-dry          4455  3.039137  2.188877  0.743311
    11            Linear    cold-dry          7630  4.709625  3.700810  0.523191
    12     BayesianRidge    cold-dry          7630  4.712181  3.707832  0.522674
    13               GLM    cold-dry          7630  4.316735  3.211386  0.599426
    14            KNNReg    cold-dry          7630  4.556637  3.344402  0.553666
    15      DecisionTree    cold-dry          7630  6.074512  4.193300  0.206779
    16      RandomForest    cold-dry          7630  4.663149  3.585999  0.532555
    17      MultiLayerNN    cold-dry          7630  4.360421  3.259569  0.591278
    18       AdaBoostReg    cold-dry          7630  5.142084  4.369851  0.431605
    19  GradientBoosting    cold-dry          7630  4.333187  3.218998  0.596367
    20     HistGradBoost    cold-dry          7630  4.324665  3.185921  0.597953
    21       StackingReg    cold-dry          7630  4.249962  3.142208  0.611723
    22            Linear  cold-humid          4407  4.406748  3.428159  0.591201
    23     BayesianRidge  cold-humid          4407  4.410080  3.429801  0.590583
    24               GLM  cold-humid          4407  3.901623  2.654909  0.679547
    25            KNNReg  cold-humid          4407  4.068069  2.681930  0.651623
    26      DecisionTree  cold-humid          4407  5.461900  3.481655  0.371999
    27      RandomForest  cold-humid          4407  4.024784  2.855962  0.658997
    28      MultiLayerNN  cold-humid          4407  3.921187  2.673124  0.676326
    29       AdaBoostReg  cold-humid          4407  4.581600  3.633421  0.558117
    30  GradientBoosting  cold-humid          4407  3.782446  2.548834  0.698825
    31     HistGradBoost  cold-humid          4407  3.773103  2.556292  0.700311
    32       StackingReg  cold-humid          4407  3.768521  2.521031  0.701039
    33            Linear   hot-humid          2448  5.905150  4.762225  0.398322
    34     BayesianRidge   hot-humid          2448  5.908823  4.760646  0.397573
    35               GLM   hot-humid          2448  5.723210  4.521828  0.434827
    36            KNNReg   hot-humid          2448  6.263350  4.916116  0.323114
    37      DecisionTree   hot-humid          2448  8.153753  6.098384 -0.147143
    38      RandomForest   hot-humid          2448  5.876157  4.716240  0.404216
    39      MultiLayerNN   hot-humid          2448  5.816675  4.573660  0.416216
    40       AdaBoostReg   hot-humid          2448  6.138614  5.159494  0.349806
    41  GradientBoosting   hot-humid          2448  5.902707  4.635588  0.398820
    42     HistGradBoost   hot-humid          2448  5.683581  4.529421  0.442626
    43       StackingReg   hot-humid          2448  5.591529  4.460259  0.460535
    44            Linear     mixture         18940  4.932446  3.835373  0.522044
    45     BayesianRidge     mixture         18940  4.933760  3.836578  0.521790
    46               GLM     mixture         18940  4.316623  3.152938  0.633941
    47            KNNReg     mixture         18940  4.481054  3.161045  0.605522
    48      DecisionTree     mixture         18940  5.980522  4.061709  0.297347
    49      RandomForest     mixture         18940  4.746297  3.573038  0.557439
    50      MultiLayerNN     mixture         18940  4.363313  3.137612  0.625979
    51       AdaBoostReg     mixture         18940  5.287412  4.435142  0.450776
    52  GradientBoosting     mixture         18940  4.284598  3.093369  0.639352
    53     HistGradBoost     mixture         18940  4.263612  3.073554  0.642877
    54       StackingReg     mixture         18940  4.240002  3.046366  0.646821
    
    The optimal model's performance:
    10       StackingReg     hot-dry          4455  3.039137  2.188877  0.743311
    21       StackingReg    cold-dry          7630  4.249962  3.142208  0.611723
    32       StackingReg  cold-humid          4407  3.768521  2.521031  0.701039
    43       StackingReg   hot-humid          2448  5.591529  4.460259  0.460535
    54       StackingReg     mixture         18940  4.240002  3.046366  0.646821
    And we can also see the climate classification strategy on the input dataset is working.
    
    """
