# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:58:05 2016

@author: dyoung
"""

from math import sqrt
from scipy.stats import t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
plt.style.use('fivethirtyeight')

%matplotlib inline

quakes = pd.read_csv("quakes.csv")

#### Exercise 1
#### Create a plot of the relationship between stations and magnitude. How would you characterize the relationship? 
#### (If you see overplotting, you may want to add jitter to your points or make them transparent by playing with the 
#### alpha value.)

sns.set(style="white")
p1 = sns.stripplot(x="mag",y="stations",data=quakes,color="black",jitter=True,alpha=.4)
p1.set_xlabel("Magnitude")
p1.set_ylabel("Stations Reporting")
p1.set_title("Quakes Data Set")

#### Exercise 2
#### Before you go ahead and fit a linear model to this trend, if in fact there was no relationship between the two,
#### what would you expect the slope to be? What about the intercept?

"""
We would expect the slope of the line to be 0. The y-intercept should be ybar.
"""

#### Exercise 3
#### Ok, now go ahead and fit a linear model called m1 to the trend and add that line to the plot from exercise 1. 
#### Interpret your slope and intercept in the context of the problem.

m1 = smf.ols(formula = "stations ~ mag", data=quakes).fit()
print(m1.summary())

figure, p2 = plt.subplots(figsize=(10,10))
p2.scatter(x="mag",y="stations",data=quakes,color="black")
p2.set_xlabel("Magnitude")
p2.set_ylabel("Stations Reporting")
p2.set_title("Quakes Data Set")
rline = m1.params[0] + (m1.params[1] * quakes["mag"])
p2.plot(quakes["mag"],rline,'r--')

"""
The slope of the model suggests that with each increase of 1 magnitude unit an additional 46 stations 
detect the earthquake. The intercept () describes a useless scenario: the number of stations that 
detect a earthquake of magnitude 0.
"""
#### Exercise 4
#### Verify the way that lm() has computed your slope correctly by using R (Python) to do the calculation 
#### using the equation for the slope based on X and Y.

xdev = quakes["mag"] - quakes["mag"].mean()
ydev = quakes["stations"] - quakes["stations"].mean()
prod = xdev*ydev
ssd_mag = quakes["mag"].var() * len(quakes["mag"])
beta1 = sum(prod)/ssd_mag
beta1

"""
Estimating the regression slope manually gives the same value (46.235) as computed with ols.fit
"""

#### Exercise 5
#### Using R, calculate a 95% confidence interval for the slope of the model that you fit in exercise 3. 
#### Confirm the calculation using confint() (conf_in()).

### the regression as a function
def reg(model,x):
    return model.params[0] + (model.params[1] * x)
    
### calculate rss
m1_resid = (quakes["stations"] - reg(m1,quakes["mag"]))**2
m1_rss = sum(m1_resid)

### calculate SE beta_1^hat
se_beta1 = sqrt(m1_rss/m1.df_resid)/sqrt(ssd_mag)

### calculate confidence interval
CI = (beta1 - t.ppf(.975,df=m1.df_resid) * se_beta1, beta1 + t.ppf(.975,df=m1.df_resid) * se_beta1)
m1.conf_int()[1:]

#### Exercise 6
#### How many stations do you predict would be able to detect an earthquake of magnitude 7.0?
reg(m1,7)

"""
From the regression we should expect 143 stations to detect an earthquake of magnitude 7.0, 
however this requires us to extrapolate. Additionally, many points from about 5.0 to the maximum 
(near 6.5) are above the estimated slope (heteroskedasticity), suggesting that this is an underestimate. 
Given that the Richter scale is logarithmic, this linear model may not adequately describe the trend.
"""

#### Exercise 7
#### Questions 1 - 6 in this lab involve elements of data description, inference, and/or prediction.
#### Which was the dominant goal in each question?

"""
1. Description
2. Description
3. Inference (and description)
4. Inference (and description)
5. Inference (and description)
6. Prediction
"""

#### Exercise 9
#### Please simulate a data set that has the same number of observations as quakes. To start, generate
#### a vector of x's. You can either generate your own x's or use the exact same x's as the quakes data.

simulation = quakes[["mag","stations"]]

#### Exercise 10
#### Next, generate your y^y^'s (the value of the mean function at the observed x's). Please generate 
#### them by writing your own function of the form:

simulation["yhat"] = reg(m1,simulation["mag"])

#### Exercise 11
#### Now, generate the y's. Note that you'll need an estimate of ??2??2, for which you can use 
#### ??^2=RSS/n???2??^2=RSS/n???2. You can extract the vector of residuals with m1$res (m1.resid).

sigma2 = m1_rss/m1.df_resid
np.random.seed(76)
simulation["error"] = np.random.normal(loc=0,scale=sqrt(sigma2),size=(1000,1))
simulation["y_sim"] = simulation["yhat"] + simulation["error"]

#### Exercise 12
#### Finally, make a plot of your simulated data. How is it similar to the original data? How is it
#### different? How might you chance your model to make it more consistent with the data?

m_sim = smf.ols(formula = "y_sim ~ mag", data=simulation).fit()
print(m_sim.summary())

figure, p_sim = plt.subplots(figsize=(10,10))
p_sim.scatter(x="mag",y="y_sim",data=simulation,color="black")
p_sim.set_xlabel("Magnitude")
p_sim.set_ylabel("Stations Reporting")
p_sim.set_title("Quakes - Simulated")
sim_rline = m_sim.params[0] + (m_sim.params[1] * simulation["mag"])
p_sim.plot(simulation["mag"],rline,'r--')

"""
The simulation is similar to the original data in that it shows a positive correlation between magnitude 
and stations reporting. However, it appears to have a more equal variance in the residuals than the 
original data. By using a more flexible model (especially one that doesn't assume equal variance or 
linear shape) we may be able to better fit the data.
"""

figure, p_both = plt.subplots(figsize=(10,10))
p_both.scatter(x="mag",y="y_sim",data=simulation,color="red",label="simulated")
p_both.scatter(x="mag",y="stations",data=quakes,color="blue",label="original")
p_both.set_xlabel("Magnitude")
p_both.set_ylabel("Stations Reporting")
p_both.set_title("Quakes - Simulated/Original")
p_both.legend(loc="upper left")


simulation2 = pd.DataFrame()
simulation2["stations"] = simulation["y_sim"]
simulation2["mag"] = simulation["mag"]
simulation2["datatype"] = "simulated"
quakes["datatype"] = "original"
quakes2 = quakes[["stations","mag","datatype"]]
both = pd.concat([quakes2,simulation2])

p_sim = sns.lmplot(x="mag",y="stations",data=both,fit_reg=False,x_jitter=0.1, hue="datatype", 
                size=10)
p_sim.set_axis_labels("Magnitude","Stations Reporting")
p_sim.fig.suptitle("Quakes - Simulated vs Original", fontsize=34)
"""
This graph gives a good visualization comparing the two data sets (original y vs simulated y).
"""

