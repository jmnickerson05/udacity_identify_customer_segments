import numpy as np
# import pandas as pd
import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
# import sys, traceback

warnings.filterwarnings("ignore")

azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

class KMSegmenter:
    def __init__(self, df):
        self.feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
        self.pipe = self.__default_pipe(df)

    def __default_pipe(self, df):
        return Pipeline(steps=[('prep', self.scale_features(df)), ('pca', PCA()), ('km', KMeans())])

    def clean_data(self, df):
        """
		Perform feature trimming, re-encoding, and engineering for demographics
		data

		INPUT: Demographics DataFrame
		OUTPUT: Trimmed and cleaned demographics DataFrame
		"""
        # Put in code here to execute all main cleaning steps:
        # convert missing value codes into NaNs, ...

        self.feat_info.missing_or_unknown = self.feat_info.missing_or_unknown.str.strip('[]').str.split(',')
        missing_vals = {feat[0]: feat[1]
                        for feat in self.feat_info[['attribute', 'missing_or_unknown']].values.tolist()}

        ## FILL IN MISSING DATA ACCORDING TO FEATURE INFO DATA
        for c in df.columns:
            df.loc[df[c].isin(missing_vals[c]), c] = np.NaN

        # remove selected columns and rows, ...
        perc_missing = ((df.isnull().sum() / len(df)).sort_values(ascending=False)) * 100
        df.drop(perc_missing[perc_missing > 20].index.tolist(), axis=1, inplace=True)
        # TODO -- Remove rows as well as columns

        # select, re-encode, and engineer column values.
        # UPDATING DATA TYPES
        for col in self.feat_info[self.feat_info.type == 'categorical'].attribute.tolist():
            try:
                df[col] = df[col].astype('category')
            except KeyError:
                # IGNORE PREVIOUSLY DROPPED COLUMNS
                print(f'Ignoring {col}')
                pass

        # Return the cleaned dataframe.
        return df

    def scale_features(self, df):
        df = df.dropna()
        #     df = df._to_pandas()

        # TODO -- make new ColumnTransformer Object
        # define transformers
        cols_by_type = lambda x: [c for
                                  c in self.feat_info[self.feat_info.type == x].attribute.tolist()
                                  if c in list(df)]

        nums, ordinal = cols_by_type('numerical'), cols_by_type('ordinal')

        scaler = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(), ordinal),
                ('num_scaler', StandardScaler(), nums),
                ('num_imputer', SimpleImputer(), nums)
            ],
            remainder='drop',
            n_jobs=-1)

        scaler.fit_transform(df)

        return scaler

    def grid_search(self, df, show_only=True):
        self.pipe.fit(df)
        param_grid = {
            'pca__n_components': [5, 15, 30, 45, 64],
            "km__n_clusters": range(2, 11),
        }
        search = GridSearchCV(self.pipe, param_grid, n_jobs=-1)
        search.fit(df)
        print('\n', "Best parameter (CV score=%0.3f):" % search.best_score_)
        print('\n', search.best_params_)
        print('\n', search.best_estimator_.named_steps)

        if not show_only:
            return search.best_params_

    def fit_model_pipe(self, df):
        params = self.grid_search(df, show_only=False)
        km = KMeans(params['km__n_clusters'])
        pca = PCA(params['pca__n_components'])
        self.km = km
        self.pca = pca
        self.transformer = self.scale_features(df)

        return Pipeline(steps=[('prep', self.transformer), ('pca', pca), ('km', km)]).fit(df)
