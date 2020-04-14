# DataLook

TODO write readme...

Until then here is an example:

# this is an example of data to be fed into the DataLook object
stack = {'datasets': [
                      {'name': 'insurance_rates', 
                       'type': 'csv', 
                       'path': 'insurance.csv', 
                       'y_values': ['charges'], 
                       'drop': []},
                      {'name': 'heart_disease', 
                       'type': 'csv', 
                       'path': 'heart.csv', 
                       'y_values': ['target'], 
                       'drop': []},
                      {'name': 'IBM_hr', 
                       'type': 'csv', 
                       'path': 'IBM_hr.csv', 
                       'y_values': ['Attrition'], 
                       'drop': []},
                     ], 
         'transforms': [{'fname': 'self.categorize', 'params': {}},
                       ], 
         'models': [{'fname' : 'RandomForestRegressor', 'params' : {}}, 
                    {'fname' : 'DecisionTreeRegressor', 'params' : {}}, 
                    {'fname' : 'Lasso', 'params' : {}}, 
                    {'fname' : 'ElasticNet', 'params' : {}}, 
                    {'fname' : 'SVR', 'params' : {'kernel' : 'rbf'}},
                    {'fname' : 'SVR', 'params' : {'kernel' : 'linear'}},
                    {'fname' : 'SGDRegressor', 'params' : {}},
                    {'fname' : 'Ridge', 'params' : {}},
                    {'fname' : 'ExtraTreesRegressor', 'params' : {}},
                    {'fname' : 'AdaBoostRegressor', 'params' : {}},
                    {'fname' : 'GradientBoostingRegressor', 'params' : {}},
                    {'fname' : 'HistGradientBoostingRegressor', 'params' : {}},
                    {'fname' : 'RandomForestRegressor', 'params' : {'max_leaf_nodes' : 50}}, 
                   ], 
         'metrics': ['r2_score', 
                     'explained_variance_score', 
                     'median_absolute_error', 
                    ]}

# create a DataLook object
dl = DataLook()

# clear anything that exists - this is helpful in a notebook to not have to reload
# the stack clears anytime it is run but jis removes any unrun data
dl.clear_stack()

# the resutls continue to stay in a dataframe until cleared
dl.clear_results()

# add the stack object to the stack
dl.add_stack(stack)

# run all the transforms and models and return the results in a datframe
results = dl.process_stack()

# use this in a notebook to print the results sorted by the highest r2 score
results.sort_values(['data', 'r2_score'], ascending=False)