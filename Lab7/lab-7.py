import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as gbm
from sklearn.ensemble import GradientBoostingRegressor as gbm_reg

"""
df_head = ["letter"]
for i in range(1,17):
    df_head.append("V" + str(i))
lettersdf = pd.read_csv("letters.csv", names=df_head)
"""
# letters_train, letters_test = train_test_split(lettersdf, test_size = .25, random_state = 76)

letters_train = pd.read_csv("letters_train.csv")
letters_test = pd.read_csv("letters_test.csv")

letters_train_Y = letters_train["letter"]
letters_train_X = letters_train.drop(["letter"], axis=1)

letters_test_Y = letters_test["letter"]
letters_test_X = letters_test.drop(["letter"], axis=1)

#### Building a boosted tree

"""
Contruct a boosted tree to predict the class of the training images (the letters) based on its 16 features. This can be done 
with the `gbm()` function in the library of the same name. Look to the end of chapter 8 for an example of the implementation.
Note that we'll be performing a boosted *classification* tree. It's very similar to the boosted regression tree except the 
method of calculating a residual is adapted to the classification setting. Please use as your model parameters $B = 50$,
$\lambda = 0.1$, and $d = 1$. Note that this is computationally intensive, so it may take a minute to run. Which variable 
is found to be the most important?
"""

def implist(model,features,ascending=False):
    l = pd.DataFrame()
    l["features"] = features
    l["importance"] = model.feature_importances_
    l.sort_values(by="importance",axis=0,inplace=True,ascending=ascending)
    
    
    return l

def impplot(model,features,ascending=False,n=10):
    l = implist(model,features,ascending)
    lsub = l.iloc[0:n-1,]
    lsub.sort_values(by="importance",axis=0,inplace=True,ascending=True)
    ax = lsub.plot(kind="barh",x="features",y="importance")
    ax.set_ylabel("Features")
    ax.set_xlabel("Gini Importance")
    ax.set_title("Top " + str(n) + " Features")
    
b1 = gbm(learning_rate = 0.1, n_estimators = 50, max_depth = 1, criterion = "mse", min_samples_leaf = 10, random_state=5)
b1.fit(letters_train_X,letters_train_Y)
    
implist(b1,letters_train_X.columns)
impplot(b1,letters_train_X.columns)

"""
Variable 12 is by far the most important. The order of importances are similar to the one produced by R but not exact.
It is unclear why they are not exact. The order of the variable importances does not change here in Python when the random_state
variable changes. It does change for R when the seed is different. 

R may be measuring variable importance differently than Python (could be Gini vs Deviance)
"""

"""
a. Build a cross-tabulation of the predicted and actual letters (a 26 X 26 confusion matrix).
b. What is your misclassification rate? (the function `diag()` might be helpful)
c. What letter was most difficult to predict?
d. Are there any letter pairs that are particularly difficult to distinguish?
"""
 
# a
b1_pred = b1.predict(letters_test_X)

b1_conf_mat = pd.crosstab(b1_pred,letters_test_Y)

(b1_conf_mat.values.sum() - b1_conf_mat.values.trace())/b1_conf_mat.values.sum()

"""
.32 missclass rate
"""

# b
d = {'predicted':b1_pred,'actual':letters_test_Y}
diff = pd.DataFrame(data=d)
diff["missclass"] = np.where(diff["actual"] == diff["predicted"],0,1)

diff_group = diff.groupby("actual")
diff_group.mean().sort_values(by="missclass",ascending=False)

"""
Letter E
"""

def pair_missclass(conf_mat):
    pair_miss = pd.DataFrame()
    
    for i in range(1,len(conf_mat)):
        for j in range(0,i):
            num = conf_mat.iloc[i,j] + conf_mat.iloc[j,i]
            denom = sum(conf_mat.iloc[:,i]) + sum(conf_mat.iloc[:,j])
            
            mc_rate = num/denom
            
            d = {"Pair":pd.Series([conf_mat.index[i] + ", " + conf_mat.columns[j]]), "Missclass Rate":[mc_rate]}
            a = pd.DataFrame(d)
            
            pair_miss = pair_miss.append(a)
    
    pair_miss = pair_miss[["Pair","Missclass Rate"]]
    pair_miss.sort_values(by="Missclass Rate", ascending=False,inplace=True)
    
    return pair_miss
    
pair_miss = pair_missclass(b1_conf_mat)

# d
"""
From `pair_miss` it appears that the pair of D and B had to highest total missclassification rate at 12%. 
This seems reasonable as the two letters look similar. E actually appears quite often in the highest missclassified 
pairs list, which makes sense considering our answer to part c.**
"""

#### Slow the learning

"""
Build a second boosted tree model that uses even *slower* learners, that is, decrease $\lambda$ and increase $B$ 
somewhat to compensate (the slower the learner, the more of them we need). Pick the parameters of your choosing for this,
 but be wary of trying to fit a model with too high a $B$. You don't want to wait an hour for your model to fit.

a. How does the misclassification rate compare to the rate from you original model?
b. Are there any letter pairs that became particularly easier/more difficult to distinguish?
"""

b2 = gbm(learning_rate = 0.01, n_estimators = 100, max_depth = 1, criterion = "mse", min_samples_leaf = 10, random_state=5)
b2.fit(letters_train_X,letters_train_Y)

b2_pred = b2.predict(letters_test_X)

b2_conf_mat = pd.crosstab(b2_pred,letters_test_Y)

(b2_conf_mat.values.sum() - b2_conf_mat.values.trace())/b2_conf_mat.values.sum()

pair_miss2 = pair_missclass(b2_conf_mat)

"""
**a. The new missclassification rate is .521 which is actually significantly worse than the first boosted model.**

**b. Y and V are now the most difficult to distinguish pair. This pair did not appear the top 10 most difficult pairs in
 the previous model. D and B the previous most difficult to distinguish pair is now the 6th most difficult, but is still 
 more difficult in absolute (13% vs 10% total missclassification).**
 
Note: Order of pairs and missclassification rates slightly different in Python than in R
"""

"""
Another set of hyperparameters.
"""

b3 = gbm(learning_rate = 0.05, n_estimators = 100, max_depth = 1, criterion = "mse", min_samples_leaf = 10, random_state=5)
b3.fit(letters_train_X,letters_train_Y)

b3_pred = b3.predict(letters_test_X)

b3_conf_mat = pd.crosstab(b3_pred,letters_test_Y)

(b3_conf_mat.values.sum() - b3_conf_mat.values.trace())/b3_conf_mat.values.sum()

pair_miss3 = pair_missclass(b3_conf_mat)


### Communities and Crime
"""
Return to the Communities and Crime data set. In the last lab you added bagged trees and random forests to your model 
portfolio in trying to predict the crime level. Constructed model based on a boosted tree with parameters of your choosing. 
How does the test MSE compare to your existing models?
"""

# Import data
crime_train = pd.read_csv("crime-train.csv")
crime_test = pd.read_csv("crime-test.csv")

# Subset the data (getting rid of columns that aren't predictors). Col 127 is our response
crime_train = crime_train.replace("?",np.nan)
crime_train_X = crime_train.iloc[:,5:100]
crime_train_Y = crime_train["ViolentCrimesPerPop"]

crime_test = crime_test.replace("?",np.nan)
crime_test_X = crime_test.iloc[:,5:100]
crime_test_Y = crime_test["ViolentCrimesPerPop"]

crime_boost = gbm_reg(learning_rate = 0.1, n_estimators = 50, max_depth = 1, criterion = "mse", min_samples_leaf = 10, 
                      random_state=5)
crime_boost.fit(crime_train_X,crime_train_Y)

implist(crime_boost,crime_train_X.columns)
impplot(crime_boost,crime_train_X.columns)

crime_pred = crime_boost.predict(crime_test_X)
np.mean((crime_pred-crime_test_Y)**2)

"""
**The variables that were found to be important were very similar to the ones found by the bagging and random forest models
 (family related variables and race).**

**The previous test MSEs of the bagged model (0.0169559) and random forest (0.0171044) were very similar to this boosted 
model test MSE 0.0179**
"""

