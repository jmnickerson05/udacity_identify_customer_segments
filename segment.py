# import libraries here; add more as necessary
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
import inspect
from IPython.display import display as idisplay
# import sys, traceback
from pprint import pprint
%matplotlib inline

display = lambda df: idisplay((df._to_pandas() if 'modin.pandas' in str(type(azdias)) else df))
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None, "display.max_columns", None)

azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')


class KMSegmenter:
    def __init__(self, df):
        self.feat_info = self.__set_feat_info()
        self.pipe = self.__default_pipe(df)

    def __default_pipe(self, df):
        return Pipeline(steps=[('prep', self.scale_features(df)), ('pca', PCA()), ('km', KMeans())])

    def __set_feat_info(self):
        return pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')

    def categorize_binary_features(self, cleaned_df):
        """
        NOTE: If the dataframe hasn't been scrubbed of encoded "nulls"
        then this will miscategorize binary variables.
        """
        self.feat_info.loc[self.feat_info.attribute.isin(
            [c for c in list(cleaned_df) if cleaned_df[c].nunique() == 2]), 'type'] = 'binary'

    def cols_by_type(self, type_name, subset_of_columns=None):
        if not subset_of_columns:
            cols = self.feat_info.attribute.tolist()
        else:
            cols = subset_of_columns

        return [col for col in self.feat_info[self.feat_info.type == type_name].attribute.tolist()
            if col in cols]

    def print_types(self):
        sep = '*' * 15
        pprint({f'{sep} << {t} >> {sep}': self.cols_by_type(t) for t in self.feat_info.type.unique()})

    def clean_data(self, df):
        """
        Perform feature trimming, re-encoding, and engineering for demographics data

        INPUT: Demographics DataFrame
        OUTPUT: Trimmed and cleaned demographics DataFrame
        """

        # Put in code here to execute all main cleaning steps:
        # convert missing value codes into NaNs, ...

        # Example taken from: https://www.youtube.com/watch?v=7ZHRM0Fl2S8
        self.feat_info.missing_or_unknown = self.feat_info.missing_or_unknown.str.strip('[]').str.split(',')
        perc_missing_before_clean = ((df.isnull().sum() / len(df)).sort_values(ascending=False)) * 100

        missing_vals = {feat[0]: feat[1]
                        for feat in self.feat_info[['attribute', 'missing_or_unknown']].values.tolist()}

        ## FILL IN MISSING DATA ACCORDING TO FEATURE INFO DATA
        for c in df.columns:
            df.loc[df[c].isin(missing_vals[c]), c] = np.NaN
        perc_missing_after_clean = ((df.isnull().sum() / len(df)).sort_values(ascending=False)) * 100

        # remove selected columns and rows, ...
        df.drop(perc_missing_after_clean[perc_missing_after_clean > 20].index.tolist(), axis=1, inplace=True)

        # TODO -- Remove rows as well as columns
        # select, re-encode, and engineer column values.
        # UPDATING DATA TYPES
        for col in self.feat_info[self.feat_info.type == 'categorical'].attribute.tolist():
            try:
                df[col] = df[col].astype('category')
            except KeyError:
                # IGNORE PREVIOUSLY DROPPED COLUMNS
                print(f'"{col}" is not categorical...Ignoring...')
                pass

        remaining_cols = df.columns
        self.perc_missing = pd.concat([perc_missing_before_clean,
                                       perc_missing_after_clean], axis=1).reset_index()
        self.perc_missing.columns = ['Attributes', 'Missing Before', 'Missing After']
        self.perc_missing['Dropped'] = self.perc_missing.Attributes.apply(
            lambda x: False if x in remaining_cols else True
        )
        self.perc_missing = self.perc_missing.set_index('Attributes')
        self.perc_missing._to_pandas()
        # Return the cleaned dataframe.
        # print(df.head())
        return df

    def display_missing_data_diff(self):
        try:
            display(self.perc_missing)
        except AttributeError as e:
            print(e)
            print('WARNING: Missing data has not been removed yet. Please call "clean_data()" method first!')

    def scale_features(self, df):
        df = df.dropna()
        df = df._to_pandas() if 'modin.pandas' in str(type(azdias)) else df

        # cols_by_type = lambda x: [c for
        #                           c in self.feat_info[self.feat_info.type == x].attribute.tolist()
        #                           if c in list(df)]

        nums, ordinal = self.cols_by_type('numerical', list(df)), self.cols_by_type('ordinal', list(df))

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
        pipe = self.__default_pipe(df)
        pipe.fit(df)
        param_grid = {
            'pca__n_components': [5, 15, 30, 45, 64],
            "km__n_clusters": range(2, 11),
        }
        search = GridSearchCV(pipe, param_grid, n_jobs=-1)
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


print_src = lambda method_name: print(inspect.getsource(
    {name: data for name, data in inspect.getmembers(KMSegmenter)}[method_name])
)

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''