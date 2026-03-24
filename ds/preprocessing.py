import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go
import plotly.express as px

class MulticollinearityCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=0.7, method='pearson', cols=None, max_samples=None, verbose: int = 0, plt_width=1000, plt_height=800):
        self.cols_to_drop = []
        self.thresh = thresh
        self.method = method
        self.max_samples = max_samples
        self.verbose = verbose
        self.plt_width = plt_width
        self.plt_height = plt_height
        self.high_corr = pd.Series()
        self.corr = None
        self.cols = cols

    def fit(self, X, Y=None):
        X = pd.DataFrame(X)
        if self.cols is not None:
            X = X[self.cols]
        
        if Y is not None:
            Y = pd.DataFrame(Y)

        if self.max_samples is not None:
            if X.shape[0] > self.max_samples:
                sample_idxes = X.sample(self.max_samples, random_state=42).index
                X = X.loc[sample_idxes, :]
                if Y is not None:
                    Y = Y.loc[sample_idxes, :]

        X.columns = X.columns.astype(str)
        # Отрисовка тепловой карты корреляции и вывод талибцы с парами с абсолютными значениями корреляции > thresh
        if Y is not None:
            Y.columns = Y.columns.astype(str)
            self.corr = pd.concat([X, Y], axis=1).corr(numeric_only=True, method=self.method)
            X_corr = self.corr.drop(index=Y.columns, columns=Y.columns)
            self.Y_corr = self.corr[Y.columns].drop(index=Y.columns)
            self.Y_corr = self.Y_corr.sort_values(ascending=False, by=list(self.Y_corr.columns), key=abs)
        else:
            self.corr = X.corr(numeric_only=True, method=self.method)
            X_corr = self.corr.copy()
            
        self.matrix_rank = np.linalg.matrix_rank(X_corr)
        self.det = np.linalg.det(X_corr)
        
        corr_pairs = X_corr.stack().reset_index()
        corr_pairs.columns = ['feature 1', 'feature 2', 'correlation']
        corr_pairs = corr_pairs[corr_pairs['feature 1'] < corr_pairs['feature 2']]  # Удаление дубликатов
        self.high_corr: pd.Series = corr_pairs[corr_pairs['correlation'].abs() > self.thresh].sort_values(by='correlation', ascending=False, key=abs)

        high_corr_copy = self.high_corr.copy()
        while not high_corr_copy.empty:
            row = high_corr_copy.iloc[0, :]

            if Y is not None:
                col_to_drop = row['feature 1'] if abs(self.Y_corr.loc[row['feature 1'], :].abs().sum()) < abs(
                    self.Y_corr.loc[row['feature 2'], :].abs().sum()) else row['feature 2']
            else:
                col_to_drop = row['feature 1']

            self.cols_to_drop.append(col_to_drop)
            high_corr_copy = high_corr_copy[~high_corr_copy.isin([col_to_drop]).any(axis=1)]
            
        if self.verbose >= 1:
            print('Столбцы к удалению:', self.cols_to_drop)
        if self.verbose >= 2:
            X_corr_cleaned = X_corr.drop(index=self.cols_to_drop, columns=self.cols_to_drop)
            X_corr_cleaned_rank = np.linalg.matrix_rank(X_corr_cleaned)
            X_corr_cleaned_det = np.linalg.det(X_corr_cleaned)
            px.imshow(X_corr.round(2), width=self.plt_width, height=self.plt_height, zmin=-1, zmax=1,
                      color_continuous_scale=px.colors.diverging.BrBG, text_auto=True, title='Корреляция факторов'
                      ).update_layout(title_x=0.5, 
                                      title_subtitle_text=f'До: rank - {self.matrix_rank}/{X.shape[1]}, det={self.det:.3f}<br>После: rank - {X_corr_cleaned_rank}/{X_corr_cleaned.shape[1]}, det={X_corr_cleaned_det:.3f}',
                                      yaxis_type='category', xaxis_type='category').show()
            if Y is not None:
                if len(Y.columns) > 1:
                    px.imshow(self.Y_corr.round(2), width=self.plt_width, height=self.plt_height, zmin=-1, zmax=1, aspect='auto',
                              color_continuous_scale=px.colors.diverging.BrBG, title=f'Корреляция факторов с целевыми признаками').update_layout(title_x=0.5).show()
                else:
                    y_corr_data = self.Y_corr.iloc[:, 0]
                    y_name = y_corr_data.name
                    y_corr_data = pd.concat([y_corr_data.apply(abs), pd.Series(
                        ['Прямая' if val > 0 else 'Обратная' for val in y_corr_data], name='Корреляция', index=y_corr_data.index)], axis=1).sort_values(y_name)
                    fig = go.Figure(data=[
                        go.Bar(
                            y=y_corr_data.index,
                            x=y_corr_data[y_name],
                            marker_color=y_corr_data['Корреляция'].apply(
                                lambda x: px.colors.diverging.BrBG[-3] if x == 'Прямая' else px.colors.diverging.BrBG[2]),
                            orientation='h'
                        )
                    ])
                    fig.update_layout(width=self.plt_width, height=self.plt_height,
                                      title_text=f'Корреляция факторов с целевым признаком {y_name}', title_x=0.5, yaxis_tickvals=y_corr_data.index, yaxis_type='category')
                    fig.show()

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop)

 