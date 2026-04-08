import pandas as pd
import pytest


@pytest.fixture
def raw_train_df():
    """Minimal synthetic DataFrame mimicking raw train.csv structure."""
    n = 20
    return pd.DataFrame({
        # Identifiers / year columns (used by make_features to compute ages)
        "YrSold":        [2010] * n,
        "YearBuilt":     [2000] * n,
        "GarageYrBlt":   [2001] * n,
        "YearRemodAdd":  [2005] * n,
        # Numerical
        "LotFrontage":   [65.0] * n,
        "LotArea":       [8450] * n,
        "MasVnrArea":    [196.0] * n,
        "BsmtFinSF1":    [706] * n,
        "BsmtFinSF2":    [0] * n,
        "BsmtUnfSF":     [150] * n,
        "TotalBsmtSF":   [856] * n,
        "1stFlrSF":      [856] * n,
        "2ndFlrSF":      [854] * n,
        "LowQualFinSF":  [0] * n,
        "GrLivArea":     [1710] * n,
        "BsmtFullBath":  [1] * n,
        "BsmtHalfBath":  [0] * n,
        "FullBath":      [2] * n,
        "HalfBath":      [1] * n,
        "BedroomAbvGr":  [3] * n,
        "KitchenAbvGr":  [1] * n,
        "TotRmsAbvGrd":  [8] * n,
        "Fireplaces":    [0] * n,
        "GarageCars":    [2] * n,
        "GarageArea":    [548] * n,
        "WoodDeckSF":    [0] * n,
        "OpenPorchSF":   [61] * n,
        "EnclosedPorch": [0] * n,
        "3SsnPorch":     [0] * n,
        "ScreenPorch":   [0] * n,
        "PoolArea":      [0] * n,
        "MiscVal":       [0] * n,
        "MSSubClass":    [60] * n,
        # Categorical
        "MSZoning":      ["RL"] * n,
        "Street":        ["Pave"] * n,
        "Alley":         [None] * n,
        "LandContour":   ["Lvl"] * n,
        "Utilities":     ["AllPub"] * n,
        "LotConfig":     ["Inside"] * n,
        "Neighborhood":  ["CollgCr"] * n,
        "Condition1":    ["Norm"] * n,
        "Condition2":    ["Norm"] * n,
        "BldgType":      ["1Fam"] * n,
        "HouseStyle":    ["2Story"] * n,
        "RoofStyle":     ["Gable"] * n,
        "RoofMatl":      ["CompShg"] * n,
        "Exterior1st":   ["VinylSd"] * n,
        "Exterior2nd":   ["VinylSd"] * n,
        "MasVnrType":    ["BrkFace"] * n,
        "Foundation":    ["PConc"] * n,
        "Heating":       ["GasA"] * n,
        "CentralAir":    ["Y"] * n,
        "Electrical":    ["SBrkr"] * n,
        "GarageType":    ["Attchd"] * n,
        "GarageFinish":  ["RFn"] * n,
        "PavedDrive":    ["Y"] * n,
        "Fence":         [None] * n,
        "MiscFeature":   [None] * n,
        "SaleType":      ["WD"] * n,
        "SaleCondition": ["Normal"] * n,
        "MoSold":        [2] * n,
        # Ordinal
        "LotShape":      ["Reg"] * n,
        "LandSlope":     ["Gtl"] * n,
        "OverallQual":   [7] * n,
        "OverallCond":   [5] * n,
        "ExterQual":     ["Gd"] * n,
        "ExterCond":     ["TA"] * n,
        "BsmtQual":      ["Gd"] * n,
        "BsmtCond":      ["TA"] * n,
        "BsmtExposure":  ["No"] * n,
        "BsmtFinType1":  ["GLQ"] * n,
        "BsmtFinType2":  ["Unf"] * n,
        "HeatingQC":     ["Ex"] * n,
        "KitchenQual":   ["Gd"] * n,
        "Functional":    ["Typ"] * n,
        "FireplaceQu":   [None] * n,
        "GarageQual":    ["TA"] * n,
        "GarageCond":    ["TA"] * n,
        "PoolQC":        [None] * n,
        # Target
        "SalePrice":     [208500] * n,
    })


@pytest.fixture
def raw_test_df(raw_train_df):
    """Test split: same structure minus SalePrice."""
    return raw_train_df.drop(columns=["SalePrice"])


@pytest.fixture
def mini_train_xy(raw_train_df):
    from house_prices_ml_foundations.features.build import make_features
    X, y = make_features(raw_train_df, return_target=True)
    return X, y