import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet as EN
from sklearn.linear_model import ElasticNetCV as ENCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import train_test_split
import random


#### 1
"""
Contruct 4 new models starting with the full set of predictors, picking 4 different
values for $\lambda$, and finding the resulting lasso regression estimates. 
"""

crime_train = pd.read_csv("crime-train.csv")
crime_test = pd.read_csv("crime-test.csv")

crime_train = crime_train.replace("?",np.nan)
crime_train = crime_train.dropna(axis=1)
crime_train = crime_train.drop(["communityname"], axis=1)

crime_test = crime_test.replace("?",np.nan)
crime_test = crime_test.dropna(axis=1)
crime_test = crime_test.drop(["communityname"], axis=1)

crime_train_X = crime_train.drop(["ViolentCrimesPerPop"], axis=1)
crime_train_Y = crime_train["ViolentCrimesPerPop"]

crime_test_X = crime_test.drop(["ViolentCrimesPerPop"], axis=1)
crime_test_Y = crime_test["ViolentCrimesPerPop"]

l1 = Lasso(alpha=0, normalize=True, max_iter=10000)
l2 = Lasso(alpha=.001, normalize=True, max_iter=10000)
l3 = Lasso(alpha=.005, normalize=True, max_iter=10000)
l4 = Lasso(alpha=.0075, normalize=True, max_iter=10000)

l_list = [l1,l2,l3,l4]

for x in l_list:
    x.fit(crime_train_X,crime_train_Y)
    print(x.coef_)

#### 2
"""
Compute the training MSE for each of the four models. You can use your function in
`test-lab-3.Rmd` for this, though you may need to modify it depending on how
you had the function bring in the coefficients from the model object to make
the prediction.
"""

def MSE(models,x,y):
    mse_list = []
    for m in models:    
    
        fitted = m.predict(x)
        mse = np.mean((fitted-y)**2)
        
        mse_list.append(mse)

    
    return mse


lasso_mse = MSE(l_list,crime_train_X,crime_train_Y)

#### 3
"""
Compute test MSE for each of the four models.
"""

lasso_test_mse = MSE(l_list,crime_test_X,crime_test_Y)

#### 9
"""
Using both ridge and lasso regression models, find a good model that predicts the number of Apps each school will get.
Use cross validation to choose your lambda (alpha)
"""

# split into train/test
college = pd.read_csv("college.csv")
college["Private"] = [1 if x == "Yes" else 0 for x in college["Private"]]
college = college.rename(columns=lambda x: x.replace(".",""))

random.seed(76)
college_train, college_test = train_test_split(college, test_size = .33)

college_train_X = college_train.drop(["Apps"], axis=1)
college_train_Y = college_train["Apps"]

college_test_X = college_test.drop(["Apps"], axis=1)
college_test_Y = college_test["Apps"]

#random.seed(76)
#alp = [.001,.01,.1,1,10]
#ridgecv = RidgeCV(alphas=alp, normalize=True, store_cv_values=True)
#ridgecv.fit(college_train_X,college_train_Y)
#cv_values = pd.DataFrame(ridgecv.cv_values_)
#cv_values.columns = alp
#cv_values_means = cv_values.mean()

# ols
lm = smf.ols(formula="Apps ~ Private + Accept + Enroll + Top10perc + Top25perc + FUndergrad + PUndergrad + Outstate + \
                 RoomBoard + Books + Personal + PhD + Terminal + SFRatio + percalumni + Expend + GradRate",\
                data=college_train).fit()


# ridge cv
encv = ENCV(l1_ratio=.1,normalize=True,cv=10,max_iter=10000, random_state=76)
encv.fit(college_train_X,college_train_Y)
encv.alpha_ # best alpha

# lasso cv
lassocv = LassoCV(normalize=True, cv=10, max_iter=10000, random_state=76)
lassocv.fit(college_train_X,college_train_Y)
lassocv.alpha_ # best alpha


# return test MSE for both ridge and lasso models
MSE([lm,encv,lassocv],college_test_X,college_test_Y)

"""
The ridge model did very terribely compared to the ols and lasso model. 
The lasso model slightly outperformed the ridge model.
"""



