# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:22:20 2016

@author: dyoung
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
plt.style.use('fivethirtyeight')

#### 1. Fitting the model

#### Please write a function that takes the training data and the outputs an lm object. It should be of the form,

"""
In order to choose a set of predictors to fit our models, we began by fitting small set of predictors, each of which was
 chosen to be as independent of the others as possible. We then expanded the model by adding other predictors one at a 
 time and checking that the significance of any of the coefficients did not decrease significantly. If one variable was 
 indeed insignificant, this variable was removed and the fit re-evaluated.

Race was fitted to a polynomial because communities of racial homogeneity showed low rates of crime, as opposed to highly
 heterogeneous communities which showed relatively high crime rates. Adding a polynomial term allowed for the trend to have 
 a downward curvature, accounting for the low crime rates at the endpoints.
"""

def group_A_fit(training_data):
    crimedata = pd.read_csv(training_data)
    m1 = smf.ols(formula="ViolentCrimesPerPop ~ NumUnderPov + MalePctDivorce + PctPersDenseHous + pctUrban + racepctblack + \
        np.power(racepctblack,2) + PctKids2Par", data=crimedata).fit()
    return m1
    
m1 = group_A_fit("crime-train.csv")
print(m1.summary())

"""
All predictors are highly significant (p<0.01). The adjusted R-square is 0.643.
"""
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.scatter(x=m1.fittedvalues,y=m1.resid,color="black")
ax1.set_xlabel("Fitted Values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residuals vs Fitted")

fig2, ax2 = plt.subplots(figsize=(12,8))
fig2 = sm.graphics.influence_plot(m1, ax=ax2, criterion="cooks")

fig3, ax3 = plt.subplots(figsize=(10,10))
ax3.set_title("Normal Q-Q Plot")
fig3 = sm.graphics.qqplot(m1.resid, line='45', ax=ax3, fit=True)


"""
There appears to be some heteroskedasticity in the model given the residuals plot. The response variable cannot be 
logged due to existence of 0's. Additionally, the assumption of normality in errors seems to be invalid according to the 
normal qq plot. In terms of prediction however, these issues are not as important since only standard errors are affected.
"""

#### 2. Computing MSE

#### Please write a function that takes as input the output from group_A_fit and a data set, and returns the MSE.

def group_A_MSE(model,csv):
    data = pd.read_csv(csv)
    MSE = np.mean((model.fittedvalues - data["ViolentCrimesPerPop"])**2)
    return MSE

m1_mse = group_A_MSE(m1,"crime-train.csv")
m1_mse

#### Following function copied from: http://planspace.org/20150423-forward_selection_with_statsmodels/ and modified to use
#### AIC as selection criterion

"""
In the R version, stepAIC is used to generate a stepwise regression model that can step both forward and backwards. I found
only a forward version for Python. This algorithm is used to generate a model that is compared with the model Even and I
came up with just by judging variable importance at a high level.
"""

crime = pd.read_csv("crime-train.csv")
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model
    
crime_subset = crime.ix[:,5:100]
crime_subset["ViolentCrimesPerPop"] = crime["ViolentCrimesPerPop"]   
fwd_model = forward_selected(crime_subset,"ViolentCrimesPerPop")

print(fwd_model.model.formula)
print(fwd_model.summary())
fwd_model_mse = group_A_MSE(fwd_model,"crime-train.csv")

fwd_model.rsquared_adj
m1.rsquared_adj

fwd_model_mse
m1_mse

"""
As we can see, the forward selection model has an only marginaly better adjusted R squared valu at the cost of including a 
much larger set of predictors, each of which is much less significant than the ones we chose. The training MSE for the 
forward selection model is roughly 15% lower.
"""


