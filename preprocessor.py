import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas DataFrame")

        self.df = df.copy()
        self.removed_columns = []
        self.fill_values = {}
        self.scaler_params = {}
        self.one_hot_columns = []

    def remove_missing(self, threshold=0.5, fill_strategy='median'):

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold должен быть числом")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold должен быть между 0 и 1")
        if fill_strategy not in ['mean', 'median', 'mode']:
            raise ValueError("fill_strategy должна быть 'mean', 'median' или 'mode'")

        missing_ratio = self.df.isnull().mean()
        cols_remove = missing_ratio[missing_ratio > threshold].index.tolist()

        if cols_remove:
            self.removed_columns.extend(cols_remove)
            self.df = self.df.drop(columns=cols_remove)

        for col in self.df.columns:
            if self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if fill_strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif fill_strategy == 'median':
                        fill_value = self.df[col].median()
                    else:
                        fill_value = self.df[col].mode()[0]
                else:

                    # Для не числовых значений используем моду
                    fill_value = self.df[col].mode()[0]


                self.fill_values[col] = {'value': fill_value, 'strategy': fill_strategy}
                self.df[col] = self.df[col].fillna(fill_value)

        return self.df

    def encode_categorical(self):

        category_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if len(category_cols) == 0:
            return self.df

        for col in category_cols:
          # one-hot
            dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
            self.one_hot_columns.extend(dummies.columns.tolist())

            self.df = pd.concat([self.df, dummies], axis=1)
            self.df = self.df.drop(columns=[col])

        return self.df

    def normalize_numeric(self, method='minmax'):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        import numpy as np

        if method not in ['minmax', 'std']:
            raise ValueError("method должен быть 'minmax' или 'std'")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("Нет числовых колонок для нормализации")
            return self.df

        if method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.scaler = scaler

        return self.df

    def fit_transform(self, threshold=0.5, fill_strategy='median', method='minmax'):

        self.remove_missing(threshold, fill_strategy)
        self.encode_categorical()
        self.normalize_numeric(method)

        return self.df

    def get_info(self):

        return {
            'removed_columns': self.removed_columns,
            'fill_values': self.fill_values,
            'one_hot_columns': self.one_hot_columns,
            'scaler_params': self.scaler_params,
            'final_shape': self.df.shape
        }