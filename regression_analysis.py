import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder

def univariate_cox_regression(df, duration_col, event_col, feature):
    tmp = df.dropna(subset=[duration_col, event_col, feature], axis=0)
    if tmp[feature].dtype == 'object':
        le = LabelEncoder()
        tmp[feature] = le.fit_transform(tmp[feature])
        
    cph = CoxPHFitter()
    cph.fit(tmp[[duration_col, event_col, feature]], duration_col=duration_col, event_col=event_col)
    
    print(cph.summary)
    print(f"95% CI: {cph.confidence_intervals_}")
    print(f"p value: {cph.summary.loc[feature, 'p']}")
    
def multivariate_cox_regression(df, duration_col, event_col, features, adjust_vars=None):
    if adjust_vars is None:
        adjust_vars = []
    
    tmp = df.dropna(subset=[duration_col, event_col] + features + adjust_vars, axis=0)
    for feature in features:
        if tmp[feature].dtype == 'object':
            le = LabelEncoder()
            tmp[feature] = le.fit_transform(tmp[feature])
    cph = CoxPHFitter()
    cph.fit(tmp[[duration_col, event_col] + features + adjust_vars], duration_col=duration_col, event_col=event_col)
    
    print(cph.summary)
    print(f"95% CI: {cph.confidence_intervals_}")
    for feature in features:
        print(f"p value for {feature}: {cph.summary.loc[feature, 'p']}")

def univariate_logistic_regression(df, target, feature):
    """
    Perform univariate logistic regression analysis.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    target (str): The name of the target variable (dependent variable).
    feature (str): The name of the feature (independent variable) to include in the model.
    
    This function performs logistic regression using the specified feature and target variable.
    It handles missing values by dropping rows with NaNs in the target column.
    It adds a constant term to the feature matrix to account for the intercept in the regression model.
    The function prints the summary of the regression results, the 95% confidence intervals, and the p-value for the feature.
    """
    tmp = df.dropna(subset=target,axis=0)
    if tmp[feature].dtype == 'object':
        le = LabelEncoder()
        tmp[feature] = le.fit_transform(tmp[feature])
        
    X = tmp[[feature]].values
    X = sm.add_constant(X)  # 定数項を追加
    y = tmp[target]
    
    model = sm.Logit(y, X)
    result = model.fit()
    
    print(result.summary())
    print(f"95% CI: {result.conf_int()}")
    print(f"p value: {result.pvalues[1]}")

def multivariate_logistic_regression(df, target, features, adjust_vars=None):
    """
    Perform multivariate logistic regression analysis.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    target (str): The name of the target variable (dependent variable).
    features (list of str): List of feature names (independent variables) to include in the model.
    adjust_vars (list of str, optional): List of variables to adjust for. Default is None.
    
    This function performs logistic regression using the specified features and target variable.
    If adjust_vars is provided, these variables are included in the model to adjust for their effects.
    The function handles missing values by dropping rows with NaNs in the target and adjust_vars columns.
    It adds a constant term to the feature matrix to account for the intercept in the regression model.
    """

    if adjust_vars is None:
        adjust_vars = []
    
    tmp = df.dropna(subset=[target] + adjust_vars, axis=0)
    for feature in features:
        if tmp[feature].dtype == 'object':
            le = LabelEncoder()
            tmp[feature] = le.fit_transform(tmp[feature])
    X = tmp[features + adjust_vars].values
    X = sm.add_constant(X)  # 定数項を追加
    y = tmp[target]
    
    model = sm.Logit(y, X)
    result = model.fit()
    
    print(result.summary())
    print(f"95% CI: {result.conf_int()}")
    for i, feature in enumerate(features):
        print(f"p value for {feature}: {result.pvalues[i+1]}")