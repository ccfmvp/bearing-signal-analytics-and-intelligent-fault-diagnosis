import xgboost as xgb


def create_xgboost_classifier(num_classes=23, **kwargs):
    """创建XGBoost多分类器

    Args:
        num_classes: 分类数量
        **kwargs: 传递给XGBClassifier的额外参数

    Returns:
        XGBClassifier实例
    """
    default_params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 8,
        'random_state': 42,
        'eval_metric': 'mlogloss',
    }
    default_params.update(kwargs)
    return xgb.XGBClassifier(**default_params)
