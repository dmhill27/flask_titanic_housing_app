# features to create polynomial features from
# feeds into create_polynomials()
polynomial_features = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageArea','TotalSF','LotArea','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','WoodDeckSF','OpenPorchSF','TotalPorchSF','TotalPorchAndDeckSF','TotalSFOverallQual','TotalSFExterQual','TotalSFKitchenQual','TotalSFGarageCars','TotalSFTotalBath','TotalSFYearBuilt','TotalSFBsmtQual','TotalSFTotRmsAbvGrd','OverallQualTotalSF','OverallQualExterQual','OverallQualKitchenQual','OverallQualGarageCars','OverallQualTotalBath','OverallQualYearBuilt','OverallQualBsmtQual','OverallQualTotRmsAbvGrd','ExterQualTotalSF','ExterQualOverallQual','ExterQualKitchenQual','ExterQualGarageCars','ExterQualTotalBath','ExterQualYearBuilt','ExterQualBsmtQual','ExterQualTotRmsAbvGrd','KitchenQualTotalSF','KitchenQualOverallQual','KitchenQualExterQual','KitchenQualGarageCars','KitchenQualTotalBath','KitchenQualYearBuilt','KitchenQualBsmtQual','KitchenQualTotRmsAbvGrd','GarageCarsTotalSF','GarageCarsOverallQual','GarageCarsExterQual','GarageCarsKitchenQual','GarageCarsTotalBath','GarageCarsYearBuilt','GarageCarsBsmtQual','GarageCarsTotRmsAbvGrd','TotalBathTotalSF','TotalBathOverallQual','TotalBathExterQual','TotalBathKitchenQual','TotalBathGarageCars','TotalBathYearBuilt','TotalBathBsmtQual','TotalBathTotRmsAbvGrd','YearBuiltTotalSF','YearBuiltOverallQual','YearBuiltExterQual','YearBuiltKitchenQual','YearBuiltGarageCars','YearBuiltTotalBath','YearBuiltBsmtQual','YearBuiltTotRmsAbvGrd','BsmtQualTotalSF','BsmtQualOverallQual','BsmtQualExterQual','BsmtQualKitchenQual','BsmtQualGarageCars','BsmtQualTotalBath','BsmtQualYearBuilt','BsmtQualTotRmsAbvGrd','TotRmsAbvGrdTotalSF','TotRmsAbvGrdOverallQual','TotRmsAbvGrdExterQual','TotRmsAbvGrdKitchenQual','TotRmsAbvGrdGarageCars','TotRmsAbvGrdTotalBath','TotRmsAbvGrdYearBuilt','TotRmsAbvGrdBsmtQual']

# features which are skewed
# feeds into log_transform()
skewed_features = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'TotalSF', 'LotArea', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF','TotalPorchSF', 'TotalPorchAndDeckSF','TotalSFOverallQual', 'TotalSFExterQual', 'TotalSFKitchenQual','TotalSFGarageCars', 'TotalSFTotalBath', 'TotalSFYearBuilt','TotalSFBsmtQual', 'TotalSFTotRmsAbvGrd', 'OverallQualTotalSF','OverallQualExterQual', 'OverallQualKitchenQual','OverallQualGarageCars', 'OverallQualTotalBath', 'OverallQualYearBuilt','OverallQualTotRmsAbvGrd', 'ExterQualTotalSF', 'ExterQualOverallQual','ExterQualKitchenQual', 'ExterQualTotalBath', 'ExterQualYearBuilt','ExterQualTotRmsAbvGrd', 'KitchenQualTotalSF', 'KitchenQualOverallQual','KitchenQualExterQual', 'KitchenQualTotalBath', 'KitchenQualYearBuilt','KitchenQualTotRmsAbvGrd', 'GarageCarsTotalSF', 'GarageCarsOverallQual','GarageCarsTotalBath', 'GarageCarsYearBuilt', 'GarageCarsTotRmsAbvGrd','TotalBathTotalSF', 'TotalBathOverallQual', 'TotalBathExterQual','TotalBathKitchenQual', 'TotalBathGarageCars', 'TotalBathYearBuilt','TotalBathBsmtQual', 'TotalBathTotRmsAbvGrd', 'YearBuiltTotalSF','YearBuiltOverallQual', 'YearBuiltExterQual', 'YearBuiltKitchenQual','YearBuiltGarageCars', 'YearBuiltTotalBath', 'YearBuiltTotRmsAbvGrd','BsmtQualTotalSF', 'BsmtQualTotalBath', 'TotRmsAbvGrdTotalSF','TotRmsAbvGrdOverallQual', 'TotRmsAbvGrdExterQual','TotRmsAbvGrdKitchenQual', 'TotRmsAbvGrdGarageCars','TotRmsAbvGrdTotalBath', 'TotRmsAbvGrdYearBuilt']

# numerical features to be imputed
# feeds into impute()
impute_numerical_features = ['TotalBsmtSF', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
           'BsmtHalfBath', 'BsmtFullBath', 'MasVnrArea', 'GarageYrBlt', 'BsmtUnfSF']

# categorical features to be imputed
# feeds into impute()
impute_categorical_features = ['Fence', 'SaleType', 'Functional', 
             'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'GarageType', 'GarageFinish',
             'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',  'FireplaceQu', 
         'MiscFeature', 'Heating']

# interesting features, create interesting feature combinations among these features
# feeds into create_new_features()
interesting_features = ['TotalSF', 'OverallQual', 'ExterQual', 'KitchenQual', 'GarageCars', 
                            'TotalBath', 'YearBuilt', 'BsmtQual', 'TotRmsAbvGrd']

# final order of features that is consistent with model training in jupyter notebook
final_feature_order = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'TotalSF', 'BsmtBath', 'AboveGroundBath', 'TotalBath', 'TotalPorchSF', 'TotalPorchAndDeckSF', 'HasPorch', 'HasPool', 'HasDeck', 'TotalSFOverallQual', 'TotalSFExterQual', 'TotalSFKitchenQual', 'TotalSFGarageCars', 'TotalSFTotalBath', 'TotalSFYearBuilt', 'TotalSFBsmtQual', 'TotalSFTotRmsAbvGrd', 'OverallQualTotalSF', 'OverallQualExterQual', 'OverallQualKitchenQual', 'OverallQualGarageCars', 'OverallQualTotalBath', 'OverallQualYearBuilt', 'OverallQualBsmtQual', 'OverallQualTotRmsAbvGrd', 'ExterQualTotalSF', 'ExterQualOverallQual', 'ExterQualKitchenQual', 'ExterQualGarageCars', 'ExterQualTotalBath', 'ExterQualYearBuilt', 'ExterQualBsmtQual', 'ExterQualTotRmsAbvGrd', 'KitchenQualTotalSF', 'KitchenQualOverallQual', 'KitchenQualExterQual', 'KitchenQualGarageCars', 'KitchenQualTotalBath', 'KitchenQualYearBuilt', 'KitchenQualBsmtQual', 'KitchenQualTotRmsAbvGrd', 'GarageCarsTotalSF', 'GarageCarsOverallQual', 'GarageCarsExterQual', 'GarageCarsKitchenQual', 'GarageCarsTotalBath', 'GarageCarsYearBuilt', 'GarageCarsBsmtQual', 'GarageCarsTotRmsAbvGrd', 'TotalBathTotalSF', 'TotalBathOverallQual', 'TotalBathExterQual', 'TotalBathKitchenQual', 'TotalBathGarageCars', 'TotalBathYearBuilt', 'TotalBathBsmtQual', 'TotalBathTotRmsAbvGrd', 'YearBuiltTotalSF', 'YearBuiltOverallQual', 'YearBuiltExterQual', 'YearBuiltKitchenQual', 'YearBuiltGarageCars', 'YearBuiltTotalBath', 'YearBuiltBsmtQual', 'YearBuiltTotRmsAbvGrd', 'BsmtQualTotalSF', 'BsmtQualOverallQual', 'BsmtQualExterQual', 'BsmtQualKitchenQual', 'BsmtQualGarageCars', 'BsmtQualTotalBath', 'BsmtQualYearBuilt', 'BsmtQualTotRmsAbvGrd', 'TotRmsAbvGrdTotalSF', 'TotRmsAbvGrdOverallQual', 'TotRmsAbvGrdExterQual', 'TotRmsAbvGrdKitchenQual', 'TotRmsAbvGrdGarageCars', 'TotRmsAbvGrdTotalBath', 'TotRmsAbvGrdYearBuilt', 'TotRmsAbvGrdBsmtQual', 'MSSubClass_120', 'MSSubClass_150', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Street_Grvl', 'Street_Pave', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'CentralAir_N', 'CentralAir_Y', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_None', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_None', 'MiscFeature_Gar2', 'MiscFeature_None', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'YrSold_2006', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_None', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SeasonSold_fall', 'SeasonSold_spring', 'SeasonSold_summer', 'SeasonSold_winter', 'LotFrontage_sqrt', 'LotFrontage_squared', 'LotFrontage_cubed', 'LotFrontage_four', 'MasVnrArea_sqrt', 'MasVnrArea_squared', 'MasVnrArea_cubed', 'MasVnrArea_four', 'BsmtFinSF1_sqrt', 'BsmtFinSF1_squared', 'BsmtFinSF1_cubed', 'BsmtFinSF1_four', 'BsmtFinSF2_sqrt', 'BsmtFinSF2_squared', 'BsmtFinSF2_cubed', 'BsmtFinSF2_four', 'BsmtUnfSF_sqrt', 'BsmtUnfSF_squared', 'BsmtUnfSF_cubed', 'BsmtUnfSF_four', 'TotalBsmtSF_sqrt', 'TotalBsmtSF_squared', 'TotalBsmtSF_cubed', 'TotalBsmtSF_four', 'GarageArea_sqrt', 'GarageArea_squared', 'GarageArea_cubed', 'GarageArea_four', 'TotalSF_sqrt', 'TotalSF_squared', 'TotalSF_cubed', 'TotalSF_four', 'LotArea_sqrt', 'LotArea_squared', 'LotArea_cubed', 'LotArea_four', '1stFlrSF_sqrt', '1stFlrSF_squared', '1stFlrSF_cubed', '1stFlrSF_four', '2ndFlrSF_sqrt', '2ndFlrSF_squared', '2ndFlrSF_cubed', '2ndFlrSF_four', 'LowQualFinSF_sqrt', 'LowQualFinSF_squared', 'LowQualFinSF_cubed', 'LowQualFinSF_four', 'GrLivArea_sqrt', 'GrLivArea_squared', 'GrLivArea_cubed', 'GrLivArea_four', 'WoodDeckSF_sqrt', 'WoodDeckSF_squared', 'WoodDeckSF_cubed', 'WoodDeckSF_four', 'OpenPorchSF_sqrt', 'OpenPorchSF_squared', 'OpenPorchSF_cubed', 'OpenPorchSF_four', 'TotalPorchSF_sqrt', 'TotalPorchSF_squared', 'TotalPorchSF_cubed', 'TotalPorchSF_four', 'TotalPorchAndDeckSF_sqrt', 'TotalPorchAndDeckSF_squared', 'TotalPorchAndDeckSF_cubed', 'TotalPorchAndDeckSF_four', 'TotalSFOverallQual_sqrt', 'TotalSFOverallQual_squared', 'TotalSFOverallQual_cubed', 'TotalSFOverallQual_four', 'TotalSFExterQual_sqrt', 'TotalSFExterQual_squared', 'TotalSFExterQual_cubed', 'TotalSFExterQual_four', 'TotalSFKitchenQual_sqrt', 'TotalSFKitchenQual_squared', 'TotalSFKitchenQual_cubed', 'TotalSFKitchenQual_four', 'TotalSFGarageCars_sqrt', 'TotalSFGarageCars_squared', 'TotalSFGarageCars_cubed', 'TotalSFGarageCars_four', 'TotalSFTotalBath_sqrt', 'TotalSFTotalBath_squared', 'TotalSFTotalBath_cubed', 'TotalSFTotalBath_four', 'TotalSFYearBuilt_sqrt', 'TotalSFYearBuilt_squared', 'TotalSFYearBuilt_cubed', 'TotalSFYearBuilt_four', 'TotalSFBsmtQual_sqrt', 'TotalSFBsmtQual_squared', 'TotalSFBsmtQual_cubed', 'TotalSFBsmtQual_four', 'TotalSFTotRmsAbvGrd_sqrt', 'TotalSFTotRmsAbvGrd_squared', 'TotalSFTotRmsAbvGrd_cubed', 'TotalSFTotRmsAbvGrd_four', 'OverallQualTotalSF_sqrt', 'OverallQualTotalSF_squared', 'OverallQualTotalSF_cubed', 'OverallQualTotalSF_four', 'OverallQualExterQual_sqrt', 'OverallQualExterQual_squared', 'OverallQualExterQual_cubed', 'OverallQualExterQual_four', 'OverallQualKitchenQual_sqrt', 'OverallQualKitchenQual_squared', 'OverallQualKitchenQual_cubed', 'OverallQualKitchenQual_four', 'OverallQualGarageCars_sqrt', 'OverallQualGarageCars_squared', 'OverallQualGarageCars_cubed', 'OverallQualGarageCars_four', 'OverallQualTotalBath_sqrt', 'OverallQualTotalBath_squared', 'OverallQualTotalBath_cubed', 'OverallQualTotalBath_four', 'OverallQualYearBuilt_sqrt', 'OverallQualYearBuilt_squared', 'OverallQualYearBuilt_cubed', 'OverallQualYearBuilt_four', 'OverallQualBsmtQual_sqrt', 'OverallQualBsmtQual_squared', 'OverallQualBsmtQual_cubed', 'OverallQualBsmtQual_four', 'OverallQualTotRmsAbvGrd_sqrt', 'OverallQualTotRmsAbvGrd_squared', 'OverallQualTotRmsAbvGrd_cubed', 'OverallQualTotRmsAbvGrd_four', 'ExterQualTotalSF_sqrt', 'ExterQualTotalSF_squared', 'ExterQualTotalSF_cubed', 'ExterQualTotalSF_four', 'ExterQualOverallQual_sqrt', 'ExterQualOverallQual_squared', 'ExterQualOverallQual_cubed', 'ExterQualOverallQual_four', 'ExterQualKitchenQual_sqrt', 'ExterQualKitchenQual_squared', 'ExterQualKitchenQual_cubed', 'ExterQualKitchenQual_four', 'ExterQualGarageCars_sqrt', 'ExterQualGarageCars_squared', 'ExterQualGarageCars_cubed', 'ExterQualGarageCars_four', 'ExterQualTotalBath_sqrt', 'ExterQualTotalBath_squared', 'ExterQualTotalBath_cubed', 'ExterQualTotalBath_four', 'ExterQualYearBuilt_sqrt', 'ExterQualYearBuilt_squared', 'ExterQualYearBuilt_cubed', 'ExterQualYearBuilt_four', 'ExterQualBsmtQual_sqrt', 'ExterQualBsmtQual_squared', 'ExterQualBsmtQual_cubed', 'ExterQualBsmtQual_four', 'ExterQualTotRmsAbvGrd_sqrt', 'ExterQualTotRmsAbvGrd_squared', 'ExterQualTotRmsAbvGrd_cubed', 'ExterQualTotRmsAbvGrd_four', 'KitchenQualTotalSF_sqrt', 'KitchenQualTotalSF_squared', 'KitchenQualTotalSF_cubed', 'KitchenQualTotalSF_four', 'KitchenQualOverallQual_sqrt', 'KitchenQualOverallQual_squared', 'KitchenQualOverallQual_cubed', 'KitchenQualOverallQual_four', 'KitchenQualExterQual_sqrt', 'KitchenQualExterQual_squared', 'KitchenQualExterQual_cubed', 'KitchenQualExterQual_four', 'KitchenQualGarageCars_sqrt', 'KitchenQualGarageCars_squared', 'KitchenQualGarageCars_cubed', 'KitchenQualGarageCars_four', 'KitchenQualTotalBath_sqrt', 'KitchenQualTotalBath_squared', 'KitchenQualTotalBath_cubed', 'KitchenQualTotalBath_four', 'KitchenQualYearBuilt_sqrt', 'KitchenQualYearBuilt_squared', 'KitchenQualYearBuilt_cubed', 'KitchenQualYearBuilt_four', 'KitchenQualBsmtQual_sqrt', 'KitchenQualBsmtQual_squared', 'KitchenQualBsmtQual_cubed', 'KitchenQualBsmtQual_four', 'KitchenQualTotRmsAbvGrd_sqrt', 'KitchenQualTotRmsAbvGrd_squared', 'KitchenQualTotRmsAbvGrd_cubed', 'KitchenQualTotRmsAbvGrd_four', 'GarageCarsTotalSF_sqrt', 'GarageCarsTotalSF_squared', 'GarageCarsTotalSF_cubed', 'GarageCarsTotalSF_four', 'GarageCarsOverallQual_sqrt', 'GarageCarsOverallQual_squared', 'GarageCarsOverallQual_cubed', 'GarageCarsOverallQual_four', 'GarageCarsExterQual_sqrt', 'GarageCarsExterQual_squared', 'GarageCarsExterQual_cubed', 'GarageCarsExterQual_four', 'GarageCarsKitchenQual_sqrt', 'GarageCarsKitchenQual_squared', 'GarageCarsKitchenQual_cubed', 'GarageCarsKitchenQual_four', 'GarageCarsTotalBath_sqrt', 'GarageCarsTotalBath_squared', 'GarageCarsTotalBath_cubed', 'GarageCarsTotalBath_four', 'GarageCarsYearBuilt_sqrt', 'GarageCarsYearBuilt_squared', 'GarageCarsYearBuilt_cubed', 'GarageCarsYearBuilt_four', 'GarageCarsBsmtQual_sqrt', 'GarageCarsBsmtQual_squared', 'GarageCarsBsmtQual_cubed', 'GarageCarsBsmtQual_four', 'GarageCarsTotRmsAbvGrd_sqrt', 'GarageCarsTotRmsAbvGrd_squared', 'GarageCarsTotRmsAbvGrd_cubed', 'GarageCarsTotRmsAbvGrd_four', 'TotalBathTotalSF_sqrt', 'TotalBathTotalSF_squared', 'TotalBathTotalSF_cubed', 'TotalBathTotalSF_four', 'TotalBathOverallQual_sqrt', 'TotalBathOverallQual_squared', 'TotalBathOverallQual_cubed', 'TotalBathOverallQual_four', 'TotalBathExterQual_sqrt', 'TotalBathExterQual_squared', 'TotalBathExterQual_cubed', 'TotalBathExterQual_four', 'TotalBathKitchenQual_sqrt', 'TotalBathKitchenQual_squared', 'TotalBathKitchenQual_cubed', 'TotalBathKitchenQual_four', 'TotalBathGarageCars_sqrt', 'TotalBathGarageCars_squared', 'TotalBathGarageCars_cubed', 'TotalBathGarageCars_four', 'TotalBathYearBuilt_sqrt', 'TotalBathYearBuilt_squared', 'TotalBathYearBuilt_cubed', 'TotalBathYearBuilt_four', 'TotalBathBsmtQual_sqrt', 'TotalBathBsmtQual_squared', 'TotalBathBsmtQual_cubed', 'TotalBathBsmtQual_four', 'TotalBathTotRmsAbvGrd_sqrt', 'TotalBathTotRmsAbvGrd_squared', 'TotalBathTotRmsAbvGrd_cubed', 'TotalBathTotRmsAbvGrd_four', 'YearBuiltTotalSF_sqrt', 'YearBuiltTotalSF_squared', 'YearBuiltTotalSF_cubed', 'YearBuiltTotalSF_four', 'YearBuiltOverallQual_sqrt', 'YearBuiltOverallQual_squared', 'YearBuiltOverallQual_cubed', 'YearBuiltOverallQual_four', 'YearBuiltExterQual_sqrt', 'YearBuiltExterQual_squared', 'YearBuiltExterQual_cubed', 'YearBuiltExterQual_four', 'YearBuiltKitchenQual_sqrt', 'YearBuiltKitchenQual_squared', 'YearBuiltKitchenQual_cubed', 'YearBuiltKitchenQual_four', 'YearBuiltGarageCars_sqrt', 'YearBuiltGarageCars_squared', 'YearBuiltGarageCars_cubed', 'YearBuiltGarageCars_four', 'YearBuiltTotalBath_sqrt', 'YearBuiltTotalBath_squared', 'YearBuiltTotalBath_cubed', 'YearBuiltTotalBath_four', 'YearBuiltBsmtQual_sqrt', 'YearBuiltBsmtQual_squared', 'YearBuiltBsmtQual_cubed', 'YearBuiltBsmtQual_four', 'YearBuiltTotRmsAbvGrd_sqrt', 'YearBuiltTotRmsAbvGrd_squared', 'YearBuiltTotRmsAbvGrd_cubed', 'YearBuiltTotRmsAbvGrd_four', 'BsmtQualTotalSF_sqrt', 'BsmtQualTotalSF_squared', 'BsmtQualTotalSF_cubed', 'BsmtQualTotalSF_four', 'BsmtQualOverallQual_sqrt', 'BsmtQualOverallQual_squared', 'BsmtQualOverallQual_cubed', 'BsmtQualOverallQual_four', 'BsmtQualExterQual_sqrt', 'BsmtQualExterQual_squared', 'BsmtQualExterQual_cubed', 'BsmtQualExterQual_four', 'BsmtQualKitchenQual_sqrt', 'BsmtQualKitchenQual_squared', 'BsmtQualKitchenQual_cubed', 'BsmtQualKitchenQual_four', 'BsmtQualGarageCars_sqrt', 'BsmtQualGarageCars_squared', 'BsmtQualGarageCars_cubed', 'BsmtQualGarageCars_four', 'BsmtQualTotalBath_sqrt', 'BsmtQualTotalBath_squared', 'BsmtQualTotalBath_cubed', 'BsmtQualTotalBath_four', 'BsmtQualYearBuilt_sqrt', 'BsmtQualYearBuilt_squared', 'BsmtQualYearBuilt_cubed', 'BsmtQualYearBuilt_four', 'BsmtQualTotRmsAbvGrd_sqrt', 'BsmtQualTotRmsAbvGrd_squared', 'BsmtQualTotRmsAbvGrd_cubed', 'BsmtQualTotRmsAbvGrd_four', 'TotRmsAbvGrdTotalSF_sqrt', 'TotRmsAbvGrdTotalSF_squared', 'TotRmsAbvGrdTotalSF_cubed', 'TotRmsAbvGrdTotalSF_four', 'TotRmsAbvGrdOverallQual_sqrt', 'TotRmsAbvGrdOverallQual_squared', 'TotRmsAbvGrdOverallQual_cubed', 'TotRmsAbvGrdOverallQual_four', 'TotRmsAbvGrdExterQual_sqrt', 'TotRmsAbvGrdExterQual_squared', 'TotRmsAbvGrdExterQual_cubed', 'TotRmsAbvGrdExterQual_four', 'TotRmsAbvGrdKitchenQual_sqrt', 'TotRmsAbvGrdKitchenQual_squared', 'TotRmsAbvGrdKitchenQual_cubed', 'TotRmsAbvGrdKitchenQual_four', 'TotRmsAbvGrdGarageCars_sqrt', 'TotRmsAbvGrdGarageCars_squared', 'TotRmsAbvGrdGarageCars_cubed', 'TotRmsAbvGrdGarageCars_four', 'TotRmsAbvGrdTotalBath_sqrt', 'TotRmsAbvGrdTotalBath_squared', 'TotRmsAbvGrdTotalBath_cubed', 'TotRmsAbvGrdTotalBath_four', 'TotRmsAbvGrdYearBuilt_sqrt', 'TotRmsAbvGrdYearBuilt_squared', 'TotRmsAbvGrdYearBuilt_cubed', 'TotRmsAbvGrdYearBuilt_four', 'TotRmsAbvGrdBsmtQual_sqrt', 'TotRmsAbvGrdBsmtQual_squared', 'TotRmsAbvGrdBsmtQual_cubed', 'TotRmsAbvGrdBsmtQual_four']