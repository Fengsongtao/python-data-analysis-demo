# 数据集划分
X_train_split, X_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2)
train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
valid_matrix = lgb.Dataset(X_val, label=y_val)
"""定义优化函数"""
def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf, 
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
    # 建立模型
    model_lgb = lgb.LGBMClassifier(boosting_type='gbdt'
                                   , bjective='binary'
                                   , metric='auc',
                                   learning_rate=0.1
                                   , n_estimators=5000,
                                   num_leaves=int(num_leaves)
                                   , max_depth=int(max_depth)
                                   , bagging_fraction=round(bagging_fraction, 2)
                                   , feature_fraction=round(feature_fraction, 2),
                                   bagging_freq=int(bagging_freq)
                                   , min_data_in_leaf=int(min_data_in_leaf),
                                   min_child_weight=min_child_weight
                                   , min_split_gain=min_split_gain,
                                   reg_lambda=reg_lambda
                                   , reg_alpha=reg_alpha
                                   , n_jobs= 8
                                  )
    
    val = cross_val_score(model_lgb, X_train_split, y_train_split, cv=5, scoring='roc_auc').mean()
    
    return val
    
    from bayes_opt import BayesianOptimization
"""定义优化参数"""
bayes_lgb = BayesianOptimization(
    rf_cv_lgb, 
    {
        'num_leaves':(10, 200),
        'max_depth':(3, 20),
        'bagging_fraction':(0.5, 1.0),
        'feature_fraction':(0.5, 1.0),
        'bagging_freq':(0, 100),
        'min_data_in_leaf':(10,100),
        'min_child_weight':(0, 10),
        'min_split_gain':(0.0, 1.0),
        'reg_alpha':(0.0, 10),
        'reg_lambda':(0.0, 10),
    }
)

"""开始优化"""
bayes_lgb.maximize(n_iter=10)
