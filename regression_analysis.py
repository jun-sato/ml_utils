import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def multivariate_cox_regression(df, target, features):
    cph = CoxPHFitter()
    cph.fit(df[features + [target]], duration_col='手術日', event_col='死亡')
    cph.print_summary()
    
def univariate_logistic_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
    
def multivariate_logistic_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")

def multivariate_logistic_regression_adjusted(data, dependent_var, independent_vars, adjustment_vars=None):
    """
    data: データフレーム
    dependent_var: 目的変数のカラム名
    independent_vars: 説明変数のリスト
    adjustment_vars: 調整変数のリスト（デフォルトはNone）
    """
    # 説明変数と調整変数を結合
    if adjustment_vars:
        independent_vars += adjustment_vars
    
    # 説明変数に定数項を追加
    X = sm.add_constant(data[independent_vars])
    y = data[dependent_var]
    
    # ロジスティック回帰モデルをフィッティング
    model = sm.Logit(y, X)
    result = model.fit()
    
    return result