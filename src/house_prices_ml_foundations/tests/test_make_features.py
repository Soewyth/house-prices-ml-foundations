from house_prices_ml_foundations.features.build import make_features


def test_make_features_returns_expected_columns(raw_train_df):
    X, y = make_features(raw_train_df.copy(), return_target=True)
    assert len(X) == 20


def test_make_features_creates_expected_age_columns(raw_train_df):
    X, _ = make_features(raw_train_df.copy(), return_target=True)
    for col in ["house_age", "garage_age", "remod_age"]:
        assert col in X.columns


def test_make_features_on_test_df_without_target_does_not_crash(raw_test_df):
    X = make_features(raw_test_df.copy(), return_target=False)
    assert X is not None
    assert len(X) == 20