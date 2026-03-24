from typing import Literal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import plotly.express as px
from scipy import stats


def distribution_vis(X: pd.DataFrame, y: pd.Series | None = None, 
                     is_category:bool=False, box_plots:bool=False,
                     agg:Literal['mean', 'median']='median', dist_mode:Literal['default', 'qq']='default', qqdist='norm', title:str | None = None, width=1600, aspect_ratio:float=1.7, max_samples:int | None = None):
    """
    Визуализация распределений признаков с использованием Plotly с гистограммами, KDE и boxplot

    Параметры:
    X - DataFrame с данными
    y - Series с целевой переменной 
    is_category - являются ли столбцы в X категориальными
    title - заголовок диаграммы
    width - ширина диаграммы
    aspect_ratio - соотношение сторон подграфика (width/height)
    """
    X = X.copy()
    if y is not None: y = y.copy()
    
    if max_samples is not None:
        if X.shape[0] > max_samples:
            random_idxes = X.sample(max_samples, random_state=42).index
            X = X.loc[random_idxes, :]
            if y is not None:
                y = y.loc[random_idxes]
    
    features = list(X.columns)

    n_features = len(features)
    n_cols = min(n_features, 3)
    n_rows = math.ceil(n_features / n_cols)

    cell_width = width / n_cols
    cell_height = cell_width / aspect_ratio

    fig_width = width
    fig_height = cell_height * n_rows
    
    def inject_passes(subtitles, p=3):
        new_subtitles = []
        for i, e in enumerate(subtitles):
            if i % p == 0 and i > 0:
                new_subtitles += [''] * p  
            new_subtitles.append(e)
        return new_subtitles
    
    is_extended = box_plots and not is_category
    # Создаем субплоты с дополнительным рядом для boxplot
    rows = n_rows*2 if is_extended else n_rows

    fig = make_subplots(
        rows=rows,  # Удваиваем ряды (верхний - boxplot, нижний - гистограмма)
        cols=n_cols,
        # subplot_titles=features,
        subplot_titles=inject_passes(features, 3) if is_extended else features,
        horizontal_spacing = cell_width / fig_width * 0.2,
        vertical_spacing = cell_height / fig_height * 0.2,
        row_heights= [0.8, 0.2]*n_rows if is_extended else None # 20% высоты на boxplot, 80% на гистограмму
    )

    for i, feature in enumerate(X):
        row_prime = (i // n_cols)*2 + 1 if is_extended else (i // n_cols)+1  # Нечетные ряды (1, 3, 5...) для boxplot
        row_sec = row_prime + 1         # Четные ряды (2, 4, 6...) для гистограммы
        col = (i % n_cols) + 1
        
        x = X[feature]

        if is_extended:
            # Boxplot (верхний график)
            fig.add_trace(
                go.Box(
                    x=x,
                    name='',
                    showlegend=False,
                    marker_color='#2ca02c',
                    line_color='#2ca02c',
                    # boxpoints=False  # Не показывать точки
                ),
                row=row_sec, col=col
            )
            # Убираем оси X для boxplot (чтобы не дублировались)
            fig.update_xaxes(title_text=None, row=row_sec, col=col)
            fig.update_yaxes(title_text=None, row=row_sec, col=col)
        
        if y is None:
            if is_category:
                # Для категориальных признаков - только barplot (без boxplot)
                counts = x.value_counts().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=counts.index,
                        y=counts.values,
                        name=feature,
                        showlegend=False
                    ),
                    row=row_prime, col=col
                )
                fig.update_yaxes(title_text="Количество", row=row_prime, col=col)
            else:
                if dist_mode == 'default':
                    # Для числовых признаков - boxplot + гистограмма с KDE
                    # Гистограмма (нижний график)
                    bin_width = (x.max() - x.min()) / 30
                    fig.add_trace(
                        go.Histogram(
                            x=x,
                            name=feature,
                            xbins=dict(size=bin_width),
                            showlegend=False,
                            marker_color='#1f77b4',
                            opacity=0.7
                        ),
                        row=row_prime, col=col
                    )

                    # KDE
                    try:
                        kde = gaussian_kde(x)
                        x_kde = np.linspace(x.min(), x.max(), 500)
                        y_kde_density = kde(x_kde)
                        y_kde_counts = (y_kde_density - np.min(y_kde_density)) / (np.max(y_kde_density) -
                                                                                np.min(y_kde_density)) * max(np.histogram(x, bins=30)[0])
                    except:
                        continue

                    fig.add_trace(
                        go.Scatter(
                            x=x_kde,
                            y=y_kde_counts,
                            mode='lines',
                            line=dict(color='red', width=2),
                            hoverinfo='text',
                            text=[f'{d:.4f}' for d in y_kde_density],
                            name='KDE',
                            hovertemplate='<b>KDE</b><br>' + 'x: %{x:.2f}<br>' + 'Density: %{text}<extra></extra>',
                            showlegend=False
                        ),
                        row=row_prime, col=col
                    )
                    fig.update_xaxes(title_text=None, row=row_prime, col=col)
                    fig.update_yaxes(title_text="Количество", row=row_prime, col=col)
                elif dist_mode == 'qq':
                    qq = stats.probplot(x, dist=qqdist, sparams=(1))
                    x_qq = np.array([qq[0][0][0], qq[0][0][-1]])

                    fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers', row=row_prime, col=col, showlegend=False)
                    fig.add_scatter(x=x_qq, y=qq[1][1] + qq[1][0]*x_qq, mode='lines', row=row_prime, col=col, showlegend=False)
        else:
            if is_category:
                # Для категориальных признаков с целевой переменной
                if box_plots:
                    fig.add_traces(
                        px.box(x=x, y=y).data, rows=row_prime, cols=col
                    )
                else:
                    if agg == 'mean': 
                        agg_values = pd.concat([x, y], axis=1).groupby(feature)[y.name].mean().sort_index()
                    elif agg == 'median': 
                        agg_values = pd.concat([x, y], axis=1).groupby(feature)[y.name].median().sort_index()
                    fig.add_trace(
                        go.Bar(
                            x=agg_values.index,
                            y=agg_values.values,
                            name=feature,
                            showlegend=False
                        ),
                        row=row_prime, col=col
                    )
                fig.update_xaxes(title_text=None, row=row_prime, col=col)
                fig.update_yaxes(title_text=f"{agg} {y.name}", row=row_prime, col=col)
            else:
                # Для числовых признаков с целевой переменной - scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        marker=dict(size=4, opacity=0.5),
                        name=feature,
                        showlegend=False
                    ),
                    row=row_prime, col=col
                )
                fig.update_yaxes(title_text=y.name, row=row_prime, col=col)

            # fig.update_xaxes(title_text=feature, row=row_boxplot, col=col)

    # Обновление общего вида
    fig.update_layout(
        width=fig_width,
        height=fig_height, 
        title_text=title,
        title_x=0.5,
        margin=dict(l=50, r=50, b=50, t=50 if not title else 80),
        bargap=0.1,
        plot_bgcolor='white'
    )

    # Настройка отображения осей и сетки
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig

 