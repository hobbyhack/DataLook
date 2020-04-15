import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler
from sklearn.base import clone

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet, SGDRegressor, Ridge
from sklearn.svm import SVR

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

@dataclass
class DataLook:
    data_stack = {
        'datasets' : [],
        'transforms' : [],
        'models' : [],
        'metrics' : [],
        'scalers' : [],
    }
    
    results = []
    
    def clear_stack(self):
        """
        function to remove all items from stack
        """
        self.data_stack = {
            'datasets' : [],
            'transforms' : [],
            'models' : [],
            'metrics' : [],
            'scalers' : [],
        }
    
    def clear_results(self):
        """
        function to clear the results
        """
        self.results = []
    
    def add_result(self, dataset, transforms, model, scores, model_result, scaler):
        """
        Function to add a single result to the results list
        """
        
        transform_str = ''
        for transform in transforms:
            transform_str += f'{transform["fname"]}({transform["params"]}),'
            
        
        result = {
            'data' : dataset['name'],
            'target' : dataset['y_values'],
            'transform' : transform_str,
            'model' : model['fname'],
            'params' : model['params'],
            'runtime_sec' : model_result['runtime_sec'],
            'scaler' : scaler
        }
        
        result.update(scores)

        result['long_param'] = model_result['params']
        
        self.results.append(result)
        
    def results_to_df(self):
        """
        Function to return a dataframe with the metrics results
        """
        return(pd.DataFrame(self.results))
    
    def add_stack(self, stack):
        for dataset in stack['datasets']:
            self.data_stack['datasets'].append(dataset)
        
        for transform in stack['transforms']:
            self.data_stack['transforms'].append(transform)
            
        for model in stack['models']:
            self.data_stack['models'].append(model)
            
        for metric in stack['metrics']:
            self.data_stack['metrics'].append(metric)
        
        for scaler in stack['scalers']:
            self.data_stack['scalers'].append(scaler)

    def add_dataset(self, dataset):
        self.data_stack['datasets'].append(dataset)
    
    def add_transform(self, transform):
        self.data_stack['transforms'].append(transform)
    
    def add_model(self, model):
        self.data_stack['models'].append(model)
    
    def add_metric(self, metric):
        self.data_stack['metrics'].append(metric)
        
    def extract_data(self, dataset):
        """
        Function to extract data
        """
        if dataset['type'] == 'csv':
            df = pd.read_csv(dataset['path'])
            df.drop(dataset['drop'], axis=1, inplace=True)
        return(df)
    
    def get_x_y(self, df, y_values):
        y = df[y_values]
        X = df.drop(y_values, axis=1)
        return([y, X])
    
    def split_data(self, df, y_values):
        y = df[y_values]
        X = df.drop(y_values, axis=1)
        
        return(train_test_split(X, y, random_state = 29))
    
    def transform_data(self, df, transforms):
        """
        Function to run all the transformations
        """
        for transform in transforms:
            func = transform['fname'] + '(' + 'df=df, ' + '**transform["params"])'
            df = eval(func)
            return(df)
    
    def categorize(self, df):
        """
        Function to separate categorical data
        """
        categorical_mask = df.dtypes == object
        categorical_features = df.columns[categorical_mask].tolist()

        binary_features = []
        multiple_features = []
        for feature in categorical_features:
            if df[feature].value_counts().count() == 2:
                binary_features.append(feature)
            if df[feature].value_counts().count() > 2:
                multiple_features.append(feature)

        if len(binary_features) > 0:     
            encoder = LabelEncoder()

            for feature in binary_features:
                df[feature] = encoder.fit_transform(df[feature])

        if len(multiple_features) > 0:
            df = pd.get_dummies(df, prefix_sep='_', drop_first=True)

        return(df)
    
    def run_model(self, model_input, train_X, train_y, test_X):
        """
        Function to run models
        """
        result = {}

        start_time = time.time()

        func = model_input['fname'] + '(**model_input["params"])'
        model = eval(func)

        model.fit(train_X, train_y)
        result['predicted'] = model.predict(test_X)
        result['params'] = model
        result['runtime_sec'] = time.time() - start_time
    
        return(result)
    
    def score_results(self, metric, actual, predicted):

        func = metric + '(' + 'actual' + ', ' + 'predicted' + ')'
        score = eval(func)

        return(score)
    
    def scale_data(self, train_X, test_X, scaler):
        func = scaler + '().fit(' + 'train_X' + ')'
    #     print(func)
        
        scaler = eval(func)
        scaled_train_X = scaler.transform(train_X)
        train_X_df = pd.DataFrame(scaled_train_X, columns=train_X.columns)
        
        scaled_test_X = scaler.transform(test_X)
        test_X_df = pd.DataFrame(scaled_test_X, columns=test_X.columns)
        
        return([train_X_df, test_X_df])
        
    def process_stack(self):
        """
        Function to process everything in the stack
        """
        for dataset in self.data_stack['datasets']:
            df = self.extract_data(dataset)
            
            df = self.transform_data(df, self.data_stack['transforms'])
        
            noscale_train_X, noscale_test_X, train_y, test_y = self.split_data(df, dataset['y_values'])

            for scaler in self.data_stack['scalers']:
                
                if scaler == 'NoScale':
                    train_X, test_X = noscale_train_X, noscale_test_X
                else:
                    train_X, test_X = self.scale_data(noscale_train_X, noscale_test_X, scaler)

                for model in self.data_stack['models']:
                    result = self.run_model(model, train_X, train_y.values.ravel(), test_X)

                    scores = {}
                    for metric in self.data_stack['metrics']:
                        scores[metric] = self.score_results(metric, actual=test_y, predicted=result['predicted'])

                    self.add_result(dataset, self.data_stack['transforms'], model, scores, result, scaler)
                
        self.clear_stack()
        
        return(self.results_to_df())