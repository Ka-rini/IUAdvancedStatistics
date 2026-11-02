"""
Code for solving the advanced workbook tasks for the course
DLMDSAS01 – Advanced Statistics

Author: Karina Regner
"""

#------------------------------------Task 1------------------------------------

#import necessary libraries
from math import comb
import matplotlib.pyplot as plt
from scipy.stats import nbinom
plt.rcParams["font.family"] = "Arial"

#define values for k and p as well as 0.5% threshold:
K = 99          # number of successes
P = 0.47        # probability of success
THRESHOLD = 0.005

# define probability mass function of the negative binomial distribution, 
#following equation (1) from the advanced workbook:
"""P(M = m) = C(m+k-1, k-1) * p^k * (1-p)^m"""
def pmf(m: int, k: int = K, p: float = P) -> float:
    return comb(m + k - 1, k - 1) * p**k * (1 - p)**m

#calculate the number of meteorites before the threshold is reached:
last_above = None
for m in range(0, 1000):
    if pmf(m) >= THRESHOLD:
        last_above = m
print("Last m with P >= 0.5 %:", last_above)        
print("From m =", last_above+1, "on: P < 0.5 %")
print("probability for 139 meteorites falling into the ocean:", pmf(last_above))
print("probability for 140 meteorites falling into the ocean:", pmf(last_above+1))

# calculate the expecation value (equation 5):
expectation = K * (1 - P) / P
print (expectation)

#calculate the median until P(M <= m) >= 0.5:
total = 0
m = 0
while True:
    total = total + pmf(m)
    if total >= 0.5:
        break
    m = m + 1
print("Median:", m)

#Cross check to make sure scipy produces the same results
assert abs(pmf(111) - nbinom.pmf(111, K, P)) < 1e-9
assert abs(expectation - nbinom.mean(K, P)) < 1e-9
assert m == int(nbinom.median(K, P))

#Plot the graphic showing the probability until P(M=m)<0.5%,
#including expectation and median
xs = list(range(1, 141))
ys = [pmf(m) for m in xs]
plt.bar(xs, ys)
plt.axhline(THRESHOLD, color="grey", linestyle=":", label="0.5 % threshold")
plt.axvline(expectation, color="red", label=f"Expectation = {expectation:.2f}")
plt.axvline(m, color="green", linestyle="--", label=f"Median = {m}")

plt.xlabel("Number of meteorites m")
plt.ylabel("P(M = m)")
plt.title("Negative binomial distribution (k = 99, p = 0.47)")
plt.legend()
plt.show()


#------------------------------------Task 2------------------------------------

#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
plt.rcParams["font.family"] = "Arial"

#Calculate probability to wait between 2 and 4 hours
def S(y):
    return 67/99 * np.exp(-3*y) + 32/99 * np.exp(-4*y)

print(S(2))
print(S(4))
print(S(2) - S(4))

#Calculate the probability density function, 
#as well as mean, variance and quartiles (in hours)
#PDF
y = np.linspace(0, 5, 500)
def f(y):
    return 201/99 * np.exp(-3*y) + 128/99 * np.exp(-4*y)
#Mean
mean, _ = quad(lambda y: y * f(y), 0, np.inf)
print(mean)
#Variance
variance, _ = quad(lambda y: (y - mean)**2 * f(y), 0, np.inf)
print(variance)
#Quartiles
def area(x):
    return quad(f, 0, x)[0]
def quartil(q):
    return brentq(lambda x: area(x) - q, 0, 5)
q1, q2, q3 = quartil(0.25), quartil(0.5), quartil(0.75)
print(q1)
print(q2)
print(q3)

#Plot PDF 
y = np.linspace(0, 4, 500)
sd = np.sqrt(variance)
plt.plot(y, f(y), label="PDF f(y)")
plt.axvline(mean, color="red", label=f"Mean = {mean:.4f} h")
plt.axvspan(mean - sd, mean + sd, color="red", alpha=0.1,
            label=f"Mean ± SD (Var = {variance:.4f} h²)")
plt.axvline(q1, color="green", linestyle="--", label=f"Q1 = {q1:.3f} h")
plt.axvline(q2, color="green", linestyle="-.", label=f"Median = {q2:.3f} h")
plt.axvline(q3, color="green", linestyle=":", label=f"Q3 = {q3:.3f} h")
plt.xlabel("Waiting time y (hours)")
plt.ylabel("PDF f(y) (per hour)")
plt.title("Probability Density Function")
plt.legend()
plt.show()

#Plot histogram
minutes = np.arange(1, 241)
p_minute = S((minutes - 1)/60) - S(minutes/60)
plt.bar(minutes, p_minute, width=1, align="edge", label="P(owl in this minute)")
plt.axvline(mean*60, color="red", label=f"Mean = {mean*60:.2f} min")
plt.axvspan((mean - sd)*60, (mean + sd)*60, color="red", alpha=0.1,
            label=f"Mean ± SD (Var = {variance*3600:.2f} min²)")
plt.axvline(q1*60, color="green", linestyle="--", label=f"Q1 = {q1*60:.2f} min")
plt.axvline(q2*60, color="green", linestyle="-.", label=f"Median = {q2*60:.2f} min")
plt.axvline(q3*60, color="green", linestyle=":", label=f"Q3 = {q3*60:.2f} min")
plt.xlabel("Waiting time (minutes)")
plt.ylabel("P(hearing the owl in this minute)")
plt.title("Histogram")
plt.legend()
plt.show()


#------------------------------------Task 5------------------------------------

#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
import sympy as sp
plt.rcParams["font.family"] = "Arial"

#Set up the data observations
data = [(12, 572607908837.31), (11, 233878018541.87), (0, 0.96), (-17, 19330602203646.31),
        (-4, 10391741.13), (-10, 89191446348.16), (-13, 1207553954046.25), (-19, 54647456663003.57),
        (8, 9552064541.15), (18, 32683749296114.75), (-7, 2638878505.06), (17, 17412675439127.47),
        (13, 1165201491857.77), (10, 91307513134.35), (7, 2562104404.02), (14, 2564501641715.85),
        (-6, 600849906.93), (16, 9390289242253.86), (-15, 5384465740632.23), (19, 55191530255786.02),
        (-3, 621610.01), (20, 92453022269733.55)]
x = np.array([p[0] for p in data], dtype=float)
y = np.array([p[1] for p in data], dtype=float)

#design the Vandermonde matrix
degree = 10
X = np.vander(x, degree + 1, increasing=True)

#Calculate OLS: solves min ||y - X a||^2
alpha_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
for i, a in enumerate(alpha_ols):
    print(f"alpha_{i:2d} = {a: .4e}")

#Calculate quality of fit
rmse = np.sqrt(np.mean((y - X @ alpha_ols)**2))
print(f"RMSE: {rmse:.4e}")

#Scale x to [-1, 1] so all columns of X have comparable size
scale = np.abs(x).max()
Xs = np.vander(x / scale, 11, increasing=True)

def ridge(X_mat, y_vec, lam):
    p = X_mat.shape[1]
    penalty = np.sqrt(lam) * np.eye(p)
    penalty[0, 0] = 0.0
    X_aug = np.vstack([X_mat, penalty])
    y_aug = np.concatenate([y_vec, np.zeros(p)])
    return np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]

#Calculate Leave one out Cross Validation (LOOCV)
def loocv_rmse(lam):
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        b = ridge(Xs[keep], y[keep], lam)
        errors.append((y[i] - Xs[i] @ b) ** 2)
    return np.sqrt(np.mean(errors))

# Choose lambda by (LOOCV) - Try 1000 values of lambda between 10^-10 and 10^10 (log scale)
lambdas = np.logspace(-10, 10, 1000)
cv = [loocv_rmse(l) for l in lambdas]
best_lam = lambdas[np.argmin(cv)]
print(f"Best lambda (LOOCV): {best_lam:.4e}")


#Compare OLS (lam = 0) with ridge for a few lambdas
for lam in [0, best_lam]:
    b = ridge(Xs, y, lam)
    alpha = b / scale ** np.arange(degree + 1)
    rmse = np.sqrt(np.mean((y - Xs @ b) ** 2))
    print(f"\nlambda = {lam:.2e}: training RMSE = {rmse:.4e}, LOOCV RMSE = {loocv_rmse(lam):.4e}")
    for i, a in enumerate(alpha):
        print(f"  alpha_{i:2d} = {a: .4e}")

#model estimation
y_scale = 1e13
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*"],
    unary_operators=["square", "cube"],
    maxsize=20,
    verbosity=0,
    )
X_estimation = x.reshape(-1, 1)  
model.fit(X_estimation, y / y_scale)
function = str(sp.simplify(model.sympy() * y_scale)).replace("x0", "x")
r2 = round(model.score(X_estimation, y / y_scale), 4)
print(f"Best function: {function}  (R² = {r2})")

#Plot data for inspection and comparison to 9*x^10, OLS and ridge
x_plot = np.linspace(x.min(), x.max(), 500)
X_plot = np.vander(x_plot / scale, degree + 1, increasing=True)

b_ols = ridge(Xs, y, 0)
b_ridge = ridge(Xs, y, best_lam)

plt.scatter(x, y, color="black", zorder=3, label="Sample points")
plt.plot(x_plot, X_plot @ b_ols, label="OLS")
plt.plot(x_plot, X_plot @ b_ridge, linestyle="--", label=f"Ridge (λ = {best_lam:.4e})")
plt.plot(x_plot, 9.0109 * x_plot**10, color="grey", linestyle=":", label=r"$9.0109\,x^{10}$")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sample points and fitted models")
plt.legend()
plt.show()
