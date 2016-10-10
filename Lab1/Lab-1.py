# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 09:18:08 2016

@author: Dean
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
plt.style.use('fivethirtyeight')
%matplotlib qt

#### Exercise 1
#### How many rows are in this data set? How many columns? What do the rows and columns represent?

boston = pd.read_csv("boston.csv")
boston.shape[0]
boston.shape[1]

# 506 rows (observations) and 14 columns (variables)

#### Exercise 2
#### Make some (2-3) pairwise scatterplots of the predictors (columns) in this data set.
#### Describe your findings.

p1 = boston.plot(kind="scatter",x="crim",y="medv")
p1.set_ylabel("Median House Value (in $1000)")
p1.set_xlabel("Per-Capita Crime Rate")
p1.set_title("Boston Data Set")

"""
This scatterplot shows the relationship between per-capita crime rate and median house value 
(in 1970s which explains low value of houses). Unsurprisingly, there appears to be a negative relationship. 
Note that a lot of the observations clump to the left side indicating that most of the suburbs in the 
data set are relatively safe.
"""
p2 = boston.plot(kind="scatter",x="age",y="medv")
p2.set_ylabel("Median House Value (in $1000)")
p2.set_xlabel("Prop. of Houses Built Before 1940")
p2.set_title("Boston Data Set")

"""
This scatterplot shows the relationship between the proportion of houses built before 1940 
(proxies for older suburbs) and median house value. Again it is unsurprising that older sururbs with 
older houses have lower median values. Many of the observations clump to the right 
(100% of houses were built before 1940). Boston is a very old city so this is unsurprising as well.
"""

#### Exercise 3
#### Are any of the predictors associated with per capita crime rate? If so, explain the relationship.

m1 = smf.ols(formula = "crim~zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black \
                + lstat + medv", data=boston
            ).fit()
print(m1.summary())

"""
The predictors that are statistically significantly associated with per-capita crime rate at the 
alpha = .05 level are proportion of residential zoning (+), weighted mean distance to five 
Boston employment centers (-), accessibility to radial highways (+), proportion of blacks (-), 
and median house value (-).

The most economically significant variable of those five is distance to employment centers, 
lowering per-capita crime rate with a coefficient of nearly -1%. This relationship can be speculated 
to be caused by the fact that employed people live near their work and don't need to commit crime to survive.
"""

#### Exercise 4
#### Are there any suburbs of Boston that appear to have particularly high crime rates? 
#### Tax rate? Pupil-teacher ratios? Comment on the range of each predictor.


p3 = boston["crim"].plot.hist(bins=30)
p3.set_xlabel("Per Capita Crime Rate")
p3.set_title("Boston Data Set")

boston[boston["crim"] > 25].shape[0]

"""
The vast majority of suburbs have per-capita crime rates under 25. There are `r high.crime` of 506 suburbs
 that have extreme crime rates (over 25 per capita) but they are not the norm.
"""

p4 = boston["tax"].plot.hist(bins=30)
p4.set_xlabel("Property Tax Rate")
p4.set_title("Boston Data Set")

boston[boston["tax"] > 600].shape[0]

"""
The vast majority of suburbs have a property tax rate between 1% and 5%. However, there are a 
significant amount of suburbs (137) that are taxed nearly 7%. There is nothing in between 
5% and 6.5% interestingly.
"""

p5 = boston["ptratio"].plot.hist(bins=30)
p5.set_xlabel("Pupil-Teacher Ratio")
p5.set_title("Boston Data Set")

boston[boston["ptratio"] == 20.2].shape[0]

"""
The pupil-teacher ratio appears to mostly be distributed uniformly across a range of 12.5 to 22.5. The one major exception 
is that at the 20-21 pupil-teacher ratio bin, there is a significant spike. Upon further investigation, there are 
140 suburbs with exactly 20.2 pupil-teacher ratio. I suspect this may be due to some policy that caps class sizes
 in certain schools.
"""

#### Exercise 5
#### How many of the suburbs in this data set bound the Charles river?

s = boston["chas"].sum()

"""
There are 35 suburbs in this dataset that are bound to the Charles river.
"""

#### Exercise 6
#### What is the median pupil-teacher ratio among the towns in this data set?

med = boston["ptratio"].median()

"""
The median pupil-teacher ratio among towns in this data set is 19.05.
"""

#### Exercise 7
#### If you want to build a model to predict the average value of a home based on the other variables, 
#### what is your output/response? What is your input?

print(m1.summary())

"""
The output/response variable would be medv and the remaining variables are the input variables. According to a multiple 
regression, most predictors in the dataset do affect median house value.
"""
