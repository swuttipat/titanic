# Dataset Description
### Overview
The data has been split into two groups:
* training set (train.csv)
* test set (test.csv)

**The training set** should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use *feature engineering* to create new features.

**The test set** should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include **gender_submission.csv**, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

Data Dictionary

| Variable |                 Definition                 |                       Key                      |
|:---------|:-------------------------------------------|:-----------------------------------------------|
| survival | Survival                                   | 0 = No, 1 = Yes                                |
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex      | Sex                                        |                                                |
| Age      | Age in years                               |                                                |
| sibsp    | # of siblings / spouses aboard the Titanic |                                                |
| parch    | # of parents / children aboard the Titanic |                                                |
| ticket   | Ticket number                              |                                                |
| fare     | Passenger fare                             |                                                |
| cabin    | Cabin number                               |                                                |
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

### Variable Notes
**pclass**: A proxy for socio-economic status (SES) <br>
1st = Upper <br>
2nd = Middle <br>
3rd = Lower <br>

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

source:
https://www.kaggle.com/competitions/titanic


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

numerical_features = ['age', 'fare']
categorical_features = ['pclass', 'sex', 'sibsp', 'parch', 'embarked']
removeable_features = ['passengerid', 'name', 'ticket', 'cabin']
```


```python
train = pd.read_csv('train.csv').copy()
test = pd.read_csv('test.csv').copy()
```


```python
# See Nan values
train.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
# Rename columns name into lowercase
train.columns = [x.lower() for x in train.columns]
```


```python
# Looking inside 'age' distribution
plt.hist(train.age)
plt.title('Passenger Age Distribution')
plt.show()


print('Age mean: {}'.format(train.age.mean()))
print('Age median: {}'.format(train.age.median()))
print('Age mode: {}'.format(train.age.mode()))


```


    
![png](titanic-ml_files/titanic-ml_6_0.png)
    


    Age mean: 29.69911764705882
    Age median: 28.0
    Age mode: 0    24.0
    dtype: float64
    


```python
# Looking inside 'fare' distribution
plt.hist(train.fare)
plt.title('Ticket Fare Distribution')
plt.show()

print('fare mean: {}'.format(train.fare.mean()))
print('fare median: {}'.format(train.fare.median()))
print('fare mode: {}'.format(train.fare.mode()))

```


    
![png](titanic-ml_files/titanic-ml_7_0.png)
    


    fare mean: 32.2042079685746
    fare median: 14.4542
    fare mode: 0    8.05
    dtype: float64
    


```python
# Rename columns name into lowercase
def rename_cols(df):
    df.columns = [x.lower() for x in df.columns]
    return df
```


```python
# Remove unwanted columns
def remove_unwanted_cols(df):
    df = df.drop(labels=removeable_features, axis=1)
    return df
```


```python
# Filter out outliers
def filter_outlier(df):
    # Select the columns to filter out outliers
    cols_to_filter = numerical_features

    # Calculate the quartiles and interquartile range
    Q1 = df[cols_to_filter].quantile(0.25)
    Q3 = df[cols_to_filter].quantile(0.75)
    IQR = Q3 - Q1

    # Set the lower and upper thresholds
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    # Filter the df
    df = df[~((df[cols_to_filter] < lower_threshold) | (df[cols_to_filter] > upper_threshold)).any(axis=1)]
    
    return df
```


```python
def imputer(df):
    # Missing values imputer
    from sklearn.impute import SimpleImputer

    # Create an imputer object with median strategy
    imputer = SimpleImputer(strategy='median')

    # Fit the imputer object on the age column
    imputer.fit(df[numerical_features])

    # Transform the age column
    df[numerical_features] = imputer.transform(df[numerical_features])

    return df
```


```python
# Apply One-hot encode
def one_hot(df):
    # Transform categorical features into dummies variables
    df = pd.get_dummies(data=df, columns=categorical_features, drop_first=True)
    return df
```


```python
def scaler(df):
    
    from sklearn.preprocessing import StandardScaler

    # Create standardscaler instace
    scaler = StandardScaler()

    # Fit the instace on the selected columns
    scaler.fit(df[numerical_features])

    # Transform the selected columns
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    return df
```


```python
# Performing hyperparameter tuning
def grid_search():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Load the data
    X_train, y_train = train.drop('survived', axis=1), train.survived

    # Define the models to be used
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'RandomForestClassifier': RandomForestClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        'SGDClassifier': SGDClassifier()
    }

    # Define the hyperparameters to be tuned
    hyperparameters = {
        'LogisticRegression': {'C': [0.1, 1, 10], 'penalty': ['l2']},
        'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'RandomForestClassifier': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
        'KNeighborsClassifier': {'n_neighbors': [20, 40, 80], 'weights': ['uniform', 'distance']},
        'GaussianNB': {},
        'SGDClassifier': {'loss': ['hinge', 'log'], 'penalty': ['l1', 'l2', 'elasticnet']}
    }

    # Train and tune each model using GridSearchCV
    for model_name, model in models.items():
        gs = GridSearchCV(model, hyperparameters[model_name], cv=5, verbose=0)
        gs.fit(X_train, y_train)
        models[model_name] = gs

    # Evaluate the models on the test set
    for model_name, gs in models.items():
        # Best score for each model
        print(f'{model_name} best score: {gs.best_score_:.2f}')
    print()
    
    for model_name, gs in models.items():
        # Best parameters for each model
        print(f'{model_name} best params: {gs.best_params_}')
    print()
        
    for model_name, gs in models.items():
        cv_score = cross_val_score(gs, X_train, y_train, cv=5)
        print(f"{model_name} Regression Accuracy: {cv_score.mean():.2f} (+/- {cv_score.std():.2f})")

    for model_name, gs in models.items():
        # Best score for each model
        models[model_name] = gs.best_score_   
        
    # Get the list of models and their corresponding best scores
    model_names = list(models.keys())
    best_scores = list(list(models.values()))

    # Create a bar chart
    plt.bar(model_names, best_scores)
    plt.xlabel("Model")
    plt.ylabel("Mean Test Score")
    plt.title("Model Tuning Results")
    plt.xticks(rotation=45)
    plt.ylim(0.7, 0.9)
    plt.show()
```


```python
import time
start_time = time.time()

# Load data
train = pd.read_csv('train.csv').copy()
test = pd.read_csv('test.csv').copy()

# Data cleaning and preprocessing
train = rename_cols(train)
train = remove_unwanted_cols(train)
train = filter_outlier(train)
train = imputer(train)
train = one_hot(train)
train = scaler(train)

# Performing hyperparameter tuning
grid_search()

end_time = time.time()
time_elapsed = end_time - start_time
print("Time elapsed: ", time_elapsed)
```

    LogisticRegression best score: 0.80
    SVC best score: 0.81
    RandomForestClassifier best score: 0.82
    KNeighborsClassifier best score: 0.79
    GaussianNB best score: 0.37
    SGDClassifier best score: 0.80
    
    LogisticRegression best params: {'C': 10, 'penalty': 'l2'}
    SVC best params: {'C': 1, 'kernel': 'rbf'}
    RandomForestClassifier best params: {'max_depth': 10, 'n_estimators': 100}
    KNeighborsClassifier best params: {'n_neighbors': 20, 'weights': 'uniform'}
    GaussianNB best params: {}
    SGDClassifier best params: {'loss': 'hinge', 'penalty': 'l1'}
    
    LogisticRegression Regression Accuracy: 0.79 (+/- 0.02)
    SVC Regression Accuracy: 0.81 (+/- 0.01)
    RandomForestClassifier Regression Accuracy: 0.81 (+/- 0.03)
    KNeighborsClassifier Regression Accuracy: 0.79 (+/- 0.03)
    GaussianNB Regression Accuracy: 0.37 (+/- 0.01)
    SGDClassifier Regression Accuracy: 0.76 (+/- 0.02)
    


    
![png](titanic-ml_files/titanic-ml_15_1.png)
    


    Time elapsed:  116.55597043037415
    


```python
def run_models(gen=False):
    # Model training, predict, and evaluate
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    # Load the data
    X_train, y_train = train.drop('survived', axis=1), train.survived
    X_test = test
    
    # Initialize the models
    models = {'Logistic Regression': LogisticRegression(C=10, fit_intercept=True, max_iter=100, penalty='l2'),
              'Support Vector Machine': SVC(C=1, kernel='rbf'),
              'Random Forest': RandomForestClassifier(criterion='entropy', n_estimators=50, max_depth=10, random_state=21)}

    # Iterate over the models
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
       
        if gen == False :
            pass
        else:
            # Create .csv predict result
            result_dict = {'PassengerId':pd.read_csv('test.csv').copy().PassengerId,
                           'Survived': y_pred}
            df = pd.DataFrame(result_dict)
            df.to_csv('{}.csv'.format(name),index=False)
            print("Create .csv Model: {} --> Done".format(name))
            print()
        
```


```python
# Load data
train = pd.read_csv('train.csv').copy()
test = pd.read_csv('test.csv').copy()

# Data cleaning and preprocessing
train = rename_cols(train)
train = remove_unwanted_cols(train)
train = filter_outlier(train)
train = imputer(train)
train = one_hot(train)
train = scaler(train)
test = rename_cols(test)
test = remove_unwanted_cols(test)
test = imputer(test)
test = one_hot(test)
test = scaler(test)
test = test.drop(['sibsp_8', 'parch_9'], axis=1)

# Train and elevauate the model
run_models(gen=True) # gen=True will return predicted result as .csv file
```

    Create .csv Model: Logistic Regression --> Done
    
    Create .csv Model: Support Vector Machine --> Done
    
    Create .csv Model: Random Forest --> Done
    
    
