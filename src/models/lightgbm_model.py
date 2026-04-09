import lightgbm as lgb


def create_lightgbm_classifier(num_classes=23, **kwargs):
    """创建LightGBM多分类器

    Args:
        num_classes: 分类数量
        **kwargs: 传递给LGBMClassifier的额外参数

    Returns:
        LGBMClassifier实例
    """
    default_params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 8,
        'random_state': 42,
        'verbose': -1,
    }
    default_params.update(kwargs)
    return lgb.LGBMClassifier(**default_params)
