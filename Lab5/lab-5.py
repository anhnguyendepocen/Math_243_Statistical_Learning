import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression as lm
from sklearn.utils import resample as rs
import numpy as np
import matplotlib.pyplot as plt

### Past Performance, Future Results[^1]

"""
Our data set this week looks at the relationship between US stock prices, the earnings of the corporations, and the returns on
 investment in stocks, with returns counting both changes in stock price and dividends paid to stock holders. A corporation's 
 **earnings** in a given year is its income minus its expenses. The return on an investment over a year is the fractional 
 change in its value, $(v_{t+1} - v_t)/v_t$, and the average rate of return over $k$ years is $[(v_{t+k} - v_t)/v_t]^{1/k}$.  

Specifically, our data contains the following variables:

- `Date`, with fractions of a year indicating months
- `Price` of an index of US stocks (inflation-adjusted)
- `Earnings` per share (also inflation-adjusted);
- `Earnings_10MA_back`, a ten-year moving average of earnings, looking backwards from the current date;
- `Return_cumul`, cumulative return of investing in the stock index,  from the beginning;
- `Return_10_fwd`, the average rate of return over the next 10 years from the current date.

"Returns" will refer to `Return_10_fwd` throughout.
"""

### 1. Inventing a variable
"""
a. Add a new column, `MAPE` to the data frame, which is the ratio of `Price` to `Earnings_10MA_back`. 
    Bring up the summary statistics for the new column using the `summary()` command. Why are there exactly 120 NAs?
    For ease of computing for the rest of the lab, you may want to remove all rows with any missing data.
b. Build a linear model to predict the returns of `MAPE`. What is the coefficient and it's standard error? 
    Is it significant?
c. What is the MSE of this model under five-fold CV? I recommend you go about this by adding a column to your data frame 
    indicating the randomly assigned *fold* to which every observation belongs. Then use a for-loop to fit and predict across
    each of the five folds, where you can use the appropriate data by subsetting based on the fold.
"""

# a
d = pd.read_csv("http://andrewpbray.github.io/math-243/assets/data/stock_history.csv")
d["MAPE"] = d["Price"]/d["Earnings_10MA_back"]
d["MAPE"].describe()
d = d.dropna()

"""
The first 120 rows (10 years over 12 months) did not have a 10 year moving average earnings because there's not enough 
trailing earnings data yet.
"""

# b
m1 = smf.ols(formula="Return_10_fwd ~ MAPE", data=d)
m1.fit().summary()
m1.fit().params[1]
"""
The coefficient and standard error for MAPE is -0.0046 and <0.001 respectively. It is very statistically significant.
"""

# c
sk_m1 = lm()
mse = np.abs(cross_val_score(sk_m1,d["MAPE"].reshape(1484,1),d["Return_10_fwd"], scoring="neg_mean_squared_error",cv=5))
mse.mean()


### 2. Inverting a variable
"""
a. Build a linear model to predict returns using `1/MAPE` (and nothing else). What is the coefficient and its standard error?
 Is it significant?
b. What is the CV MSE of this model? How does it compare to the previous one?
"""

#a
d["Inv_MAPE"] = 1/d["MAPE"]
m2 = smf.ols(formula="Return_10_fwd ~ Inv_MAPE",data=d)
m2.fit().summary()

"""
The coefficient and standard error for inverse MAPE is 0.9959 and 0.037 respectively. It is very statistically significant.
"""

# b
mse2 = np.abs(cross_val_score(sk_m1,d["Inv_MAPE"].reshape(1484,1),d["Return_10_fwd"], scoring="neg_mean_squared_error",cv=5))
mse2.mean()

"""
The two average MSEs from cross-validations are virtually identical (0.0025 vs 0.0023)
"""

### 3. A simple model
"""
A simple-minded model says that the expected returns over the next ten years should be exactly equal to `1/MAPE`.
"""

errors = (d["Return_10_fwd"] - d["Inv_MAPE"])**2
errors.mean()

### 4. Is simple sufficient?
"""
The model that we fit in no. 2 is very similar to the simple-minded model. Let's compare the similarity in these models. 
We could go about this in two ways. We could *simulate* from the simple-minded model many times and fit a model of the same
 form as no. 2 to each one to see if our observed slope in no. 2 is probable under the simple-minded model. We could also 
 *bootstrap* the data set many times, fitting model 2 each time, then see where the simple-minded model lays in that 
 distribution. Since we already have practiced with simulation, let's do the bootstrap method.

a. Form the bootstrap distribution for the slope of `1/MAPE` (slides from last class may be helpful).
 Plot this distribution with the paramter of interest (the slope corresponding to the simple-minded model) indicated by a
 vertical line.
b. What is the approximate 95% bootstrap confidence interval for the slope? How does this interval compare to the one
 returned by running `confint()` on your model object from question 2? Please explain any difference you've found.
"""

# a

def bootstrap(data,model,x,y,n_resamples):
    betas = np.zeros((5000,1))   
    
    for i in range(0,n_resamples):
        bs_d = rs(data)
        
        betas[i] = model.fit(bs_d[x].reshape(len(d),1),bs_d[y]).coef_[0]
    
    return betas
    
bs_betas = bootstrap(d,sk_m1,"Inv_MAPE","Return_10_fwd",5000)
bs_betas.mean()
bs_betas.std()

figure, ax = plt.subplots(figsize=(10,10))
ax.hist(bs_betas, bins=30)
ax.axvline(1,color="r",linestyle="dashed",linewidth=2)

#b

np.percentile(bs_betas,2.5)
np.percentile(bs_betas,97.5)
m2.fit().conf_int()
"""
Bootstrapped confidence interval for betas is [0.937,1.058] compared to the confidence interval from question 2 of
 [.924,1.068].The parametric confidence interval for the betas is slightly more uncertain (wide) than the bootstrap
 confidence interval. This could just be due to sampling variability since the difference is small. It could also be due 
 to the fact that one or more of the OLS assumptions have been violated (such as consistent variance).
"""

### 5. One big happy plot
"""
For this problem, you need to only include one plot and one paragraph of writing. Also, in this problem, take "line" 
to mean "straight or curved line" as appropriate, and be sure to plot actual lines and not disconnected points.

a. Make a scatterplot of the returns against `MAPE`.
b. Add two lines showing the predictions from the models you fit in problems 1 and 2.
c. Add a line showing the predictions from the simple-minded model from problem 3.
"""
r1 = m1.fit().params[0] + m1.fit().params[1] * d["MAPE"]
r2 = m2.fit().params[0] + m2.fit().params[1] * d["Inv_MAPE"]

figure, ax = plt.subplots(figsize=(10,10))
ax.scatter(d["MAPE"],d["Return_10_fwd"], color="pink", label="")
ax.plot(d["MAPE"],r1,color="red", label="MAPE")
ax.plot(d["MAPE"],r2,color="blue", label="Inverse MAPE")
ax.plot(d["MAPE"],d["Inv_MAPE"],color="green", label="Simple")
ax.legend()
ax.set_xlabel("MAPE")
ax.set_ylabel("Return")
ax.set_title("Correlation Between MAPE and Return")
        
### The big picture
"""
a. **Cross-validation for model selection**: using CV MSE, which model would you select to make predictions of returns? 
Looking at the plot in question 5, does this seem like a good model? What are it's strengths and weaknesses for prediction?
b. **Bootstrapping for uncertainty estimation**: based on your bootstrapping procedure for the slope of the linear model
 using `1/MAPE` as a predictor, is the simple-minded model a plausible model given our data?

a - Based on the CV MSE, I would use the second model to predict returns given it has the lowest MSE. It appears to be a 
decent fit according to the graph. However, the first model appears to do better for higher values of MAPE since that 
section appears to have a more linear relationship.

b - Yes, the simple minded model uses a beta that is contained in the bootstrapped confidence interval.
"""

"""
 Based on a homework of Cosma Shalizi at CMU.
"""

