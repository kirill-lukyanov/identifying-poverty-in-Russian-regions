from collections import defaultdict
import json
import os
import time
import plotly.express as px
import pandas as pd


class ABLogger():
    def __init__(self, path: str):
        self.path = path

    def write(self, result, A: str = '', B: str = '', mode='w'):
        log = {'A': A, 'B': B, 'result': result}
        if mode == 'w':
            logs = []
        elif mode == 'a':
            logs = self.read() or []
        else:
            raise ValueError(f'Неизвестное значение mode={mode}')
        logs.append(log)
        with open(self.path, mode='w', encoding='utf8') as f:
            f.write(json.dumps(logs))

    def read(self):
        if os.path.exists(self.path):
            with open(self.path, mode='r', encoding='utf8') as f:
                return json.loads(f.read())
        return None
    
    def remove(self, idx):
        logs = self.read()
        if logs:
            del logs[idx]
            with open(self.path, mode='w', encoding='utf8') as f:
                f.write(json.dumps(logs))


class ModelLogger():
    def __init__(self, samples, mtrcs, fit_sample='train', y_trans=None, verbose=0):
        self.logs = []
        self.samples = samples
        self.mtrcs = mtrcs
        self.fit_sample = fit_sample
        self.y_trans = y_trans
        self.verbose = verbose

    def log(self, model, model_name, samples=None):
        if samples is None:
            samples = self.samples

        start = time.time()
        model.fit(samples[self.fit_sample][0], samples[self.fit_sample][1])
        fit_time = time.time() - start
        log = defaultdict(lambda: defaultdict(dict))
        log['model_name'] = model_name
        log['fit_time'] = fit_time
        predict_time_sum = 0
        for sample_name, (X, y) in samples.items():
            start = time.time()
            y_pred = model.predict(X)
            predict_time_sum += time.time() - start
            if self.y_trans is not None:
                y_pred = self.y_trans(y_pred)
                y = self.y_trans(y)
            for mtrc, mtrc_fn in self.mtrcs.items():
                log['samples'][sample_name][mtrc] = mtrc_fn(y, y_pred)
        log['predict_time_mean'] = predict_time_sum / len(samples)
        self.logs.append(log)
        if self.verbose >= 1:
            print(f'{log['model_name']}:')
            print(f'fit time - {log['fit_time']:.3f}, predict time - {log['predict_time_mean']:.3f}')
            print(pd.DataFrame(log['samples']).T.round(3).to_markdown(tablefmt='github'))
        if self.verbose >= 2:
            px.bar(log['samples'], barmode='group',
                   title=f'{log['model_name']}',
                   subtitle=f'Время обучения - {log['fit_time']:.3f} с., время предсказания - {log['predict_time_mean']:.3f}',
                   labels={'index': 'Метрика', 'value': 'Значение'}).show()
