import pandas as pd
import numpy as np
import pickle

class PreProcess_Data():
  def __init__(self, data):
    self.data = data

  def log_transform(self, cols):
    self.data[cols] = np.log1p(self.data[cols])
  
  def impute(self, num_cols_impute, cat_cols_impute, imputer_median, imputer_mode):
    self.data['YrSold'] = self.data['YrSold'].apply(lambda x: str(x))
    self.data['MSSubClass'] = self.data['MSSubClass'].apply(lambda x: str(x))

    #numerical features where a missing value indicates a value of 0
    self.data[num_cols_impute] = self.data[num_cols_impute].fillna(0)

    #categorical features where a missing value indicates a value of 'None'
    self.data[cat_cols_impute] = self.data[cat_cols_impute].fillna('None')

    data_num = self.data[self.data.select_dtypes(exclude='object').columns]
    data_cat = self.data[self.data.select_dtypes(include='object').columns]

    num_cols = data_num.columns
    cat_cols = data_cat.columns

    data_num = pd.DataFrame(imputer_median.transform(data_num))
    data_cat = pd.DataFrame(imputer_mode.transform(data_cat))
    data_num.columns = num_cols
    data_cat.columns = cat_cols

    self.data[num_cols] = data_num
    self.data[cat_cols] = data_cat
    #self.data = pd.concat([data_num,data_cat], axis=1)
  
  def create_ordinals(self):
    print(self.data['YrSold'].value_counts())
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond', 'PoolQC']:
      self.data = self.data.replace({col: {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}})

    for col in['BsmtFinType1', 'BsmtFinType2']:
      self.data = self.data.replace({col: {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}})

    self.data = self.data.replace({'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
                          'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
                          'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
                           'Electrical': {'None': 0, 'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5},
                           'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5,
                                        'Min2': 6, 'Min1': 7, 'Typ': 8}})
  def create_bins(self):
    print(self.data['YrSold'].value_counts())
    def year_built_bins(year):
      if year == 0:
        return 0
      elif year >= 2000:
        return 4
      elif year >= 1980:
        return 3
      elif year >= 1960:
        return 2
      else:
        return 1

    def season_bins(x):
      if 3 <= x <= 5:
        return 'spring'
      elif 6 <= x <= 8:
        return 'summer'
      elif 9 <= x <= 11:
        return 'fall'
      else:
        return 'winter'

    def year_remodel_bins(x):
      if x >= 2000:
        return 4
      elif x >= 1990:
        return 3
      elif x >= 1970:
        return 2
      else:
        return 1

    self.data['GarageYrBlt'] = self.data['GarageYrBlt'].apply(lambda x: year_built_bins(x))
    self.data['YearBuilt'] = self.data['YearBuilt'].apply(lambda x: year_built_bins(x))
    self.data['SeasonSold'] = self.data['MoSold'].apply(lambda x: season_bins(x))
    self.data['YearRemodAdd'] = self.data['YearRemodAdd'].apply(lambda x: year_remodel_bins(x))
  
  def create_new_features(self, cols):
    self.data['TotalSF'] = self.data['TotalBsmtSF'] + self.data['GrLivArea']
    self.data['BsmtBath'] = self.data['BsmtFullBath'] + (0.5*self.data['BsmtHalfBath'])
    self.data['AboveGroundBath'] = self.data['FullBath'] + (0.5*self.data['HalfBath'])
    self.data['TotalBath'] = self.data['BsmtBath'] + self.data['AboveGroundBath']
    self.data['TotalPorchSF'] = self.data['EnclosedPorch'] + self.data['OpenPorchSF'] + self.data['3SsnPorch'] + self.data['ScreenPorch']
    self.data['TotalPorchAndDeckSF'] = self.data['TotalPorchSF'] + self.data['WoodDeckSF']
    self.data['HasPorch'] = self.data['TotalPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    self.data['HasPool'] = self.data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    self.data['HasDeck'] = self.data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
    self.data['BsmtUnfSF'] = self.data['BsmtUnfSF'].apply(lambda x: float(x))

    for col_one in cols:
      for col_two in cols:
        if col_one != col_two:
          new_col = col_one + col_two
          self.data[new_col] = self.data[col_one] * self.data[col_two]

  def one_hot_encode(self, encoder):
    cat_cols = self.data.select_dtypes(include='object').columns
    df_cat = self.data[cat_cols]
    self.data.drop(columns=cat_cols, inplace=True)
    df_cat_transformed = pd.DataFrame(encoder.transform(df_cat), columns=encoder.get_feature_names(list(cat_cols)))
    self.data = pd.concat([self.data, df_cat_transformed], axis=1)

  def create_polynomials(self, cols):
    def transform(df, col):
      col_sqrt = col + '_sqrt'
      df[col_sqrt] = df[col].apply(lambda x: x**(1/2))

      col_squared = col + '_squared'
      df[col_squared] = df[col].apply(lambda x: x**2)

      col_cubed = col + '_cubed'
      df[col_cubed] = df[col].apply(lambda x: x**3)

      col_poly_four = col + '_four'
      df[col_poly_four] = df[col].apply(lambda x: x**4)


    for col in cols:
      transform(self.data, col)
  def drop_columns(self, cols):
    self.data.drop(columns=cols, inplace=True)

  def get_data(self):
    return self.data