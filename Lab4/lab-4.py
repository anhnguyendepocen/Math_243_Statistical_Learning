import pandas as pd
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


#### Question 1

"""
Fit a logistic regression model for the start of civil war on all other variables except country and year (yes, 
this makes some questionable assumptions about independent observations); include a quadratic term for exports. 
Report the coefficients and their standard errors, together with R's p-values. Which ones does R say are significant at the
5% level?
"""

war = pd.read_csv("http://www.stat.cmu.edu/~cshalizi/uADA/15/hw/06/ch.csv")
war = war.ix[:,1:len(war.columns)]
war = war.dropna()

warX = war.drop(["country","year","start"], axis=1)
warX["exports^2"] = warX["exports"]**2
warX["intercept"] = 1.0
warY = war["start"]

m1 = sm.Logit(warY, warX).fit()
print(m1.summary())

"""
See table above for coefficients, SEs, and p-values.
All but dominance are significant at the 5% level, although dominance is close: p = 0.058.
"""
# Question 2
# Interpretation: All parts of this question refer to the logistic regression model you just fit.

"""
**What is the modelâs predicted probability for a civil war in India in the period beginning 1975? What probability would 
it predict for a country just like India in 1975, except that its male secondary school enrollment rate was 30 points higher?
 What probability would it predict for a country just like India in 1975, except that the ratio of commodity exports to GDP 
 was 0.1 higher?
"""

warBase = war.copy()
warBase["exports^2"] = war["exports"]**2
warBase["intercept"] = 1.0

war_India_1975 = warBase[(war.country == "India") & (war.year == 1975)]
war_India_1975 = war_India_1975.drop(["country","year","start"], axis=1)

m1.predict(war_India_1975)[0]
"""
35%
"""

war_India_1975_school = war_India_1975.copy()
war_India_1975_school["schooling"] = war_India_1975_school["schooling"] + 30

m1.predict(war_India_1975_school)[0]
"""
17%
"""

war_India_1975_exports = war_India_1975.copy()
war_India_1975_exports["exports"] = war_India_1975_exports["exports"] + 0.1
war_India_1975_exports["exports^2"] = war_India_1975_exports["exports"]**2

m1.predict(war_India_1975_exports)[0]
"""
70%
"""

"""
What is the model's predicted probability for a civil war in Nigeria in the period beginning 1965?
 What probability would it predict for a country just like Nigeria in 1965, except that its male secondary school 
 enrollment rate was 30 points higher? What probability would it predict for a country just like Nigeria in 1965, 
 except that the ratio of commodity exports to GDP was 0.1 higher?
"""

war_Nigeria_1965 = warBase[(war.country == "Nigeria") & (war.year == 1965)]
war_Nigeria_1965 = war_Nigeria_1965.drop(["country","year","start"], axis=1)

m1.predict(war_Nigeria_1965)[0]
"""
17%
"""

war_Nigeria_1965_school = war_Nigeria_1965.copy()
war_Nigeria_1965_school["schooling"] = war_Nigeria_1965_school["schooling"] + 30

m1.predict(war_Nigeria_1965_school)[0]
"""
7%
"""

war_Nigeria_1965_exports = war_Nigeria_1965.copy()
war_Nigeria_1965_exports["exports"] = war_Nigeria_1965_exports["exports"] + 0.1
war_Nigeria_1965_exports["exports^2"] = war_Nigeria_1965_exports["exports"]**2

m1.predict(war_Nigeria_1965_exports)[0]
"""
33%
"""

""" Q:
In the parts above, you changed the same predictor variables by the same amounts. If you did your calculations properly, 
the changes in predicted probabilities are not equal. Explain why not. (The reasons may or may not be the same for the 
two variables.)
"""

"""A:
The different changes are due to using logistic regression rather than linear regression: because the function does not
 have a constant slope, the response to change depends on where in the data the observations are. 
"""

# Question 3
# Build a 2x2 confusion matrix (a.k.a. A classification tableor a contigency table) which counts: the number of 
# outbreaks of civil war correctly predicted by the logistic regression; the number of civil wars not predicted by 
# the model; the number of false predictions of civil wars; and the number of correctly predicted absences of civil wars. 
# (Note that some entries in the table may be zero.)

my_log_pred = pd.DataFrame()
my_log_pred["pred"] = ["No" if x < .5 else "Yes" for x in m1.predict()]
my_log_pred["actual"] = ["No" if x == 0 else "Yes" for x in war["start"]]
conf_log = pd.crosstab(my_log_pred["pred"], my_log_pred["actual"])
conf_log

"""
*What fraction of the logistic regression's predictions are incorrect, i.e. what is the misclassification rate? 
(Note that this is if anything too kind to the model, since it's looking at predictions to the same training data set)
"""

(1/(war.shape[0])) * (conf_log.iloc[1,0] + conf_log.iloc[0,1])

"""
7%
"""


"""
Consider a foolish (?) pundit who always predicts no war. What fraction of the pundit's predictions are correct on the 
whole data set? What fraction are correct on data points where the logistic regression model also makes a prediction?
"""

conf_log.iloc[:,0].sum()/war.shape[0]

"""
93%
"""

# Question 4
# Comparison: Since this is a classification problem with only two classes, we can compare Logistic Regression right 
# along side Discriminant Analysis.

"""
Fit an LDA model using the same predictors that you used for your logistic regression model. 
What is the training misclassification rate?
"""

lda1 = LDA(solver="svd", store_covariance=True)
lda1.fit(warX,warY)

my_lda_pred = pd.DataFrame()
my_lda_pred["pred"] = ["No" if x == 0 else "Yes" for x in lda1.predict(warX)]
my_lda_pred["actual"] = ["No" if x == 0 else "Yes" for x in war["start"]]
conf_lda = pd.crosstab(my_lda_pred["pred"], my_lda_pred["actual"])
conf_lda

(1/(war.shape[0])) * (conf_lda.iloc[1,0] + conf_lda.iloc[0,1])


"""
6.69%
"""

qda1 = QDA(store_covariances=True)
qda1.fit(warX,warY)

test = qda1.predict_proba(warX)

my_qda_pred = pd.DataFrame()
my_qda_pred["pred"] = ["No" if x < .5 else "Yes" for x in qda1.predict(warX)]
my_qda_pred["actual"] = ["No" if x == 0 else "Yes" for x in war["start"]]
conf_qda = pd.crosstab(my_qda_pred["pred"], my_qda_pred["actual"])
conf_qda

(1/(war.shape[0])) * (conf_qda.iloc[1,0] + conf_qda.iloc[0,1])