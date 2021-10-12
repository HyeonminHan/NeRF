from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)


    
# numargs = skewnorm .numargs 
# a, b = 4.32, 3.18
# rv = skewnorm (a, b) 
    
# print ("RV : \n", rv)

a = -2
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')


# print("mean:", mean,"var:", var, "skew:", skew, "kurt:", kurt)
# print("skewnorm.ppf(0.01, a):", skewnorm.ppf(0.01, a))
# print("skewnorm.ppf(0.99, a):", skewnorm.ppf(0.99, a))

# print("a:", a)
# x = np.linspace(skewnorm.ppf(0.01, a),
#                 skewnorm.ppf(0.99, a), 100)
# print("x;", x)
# print("skewnorm.pdf(x, a):", skewnorm.pdf(x, a))

# ax.plot(x, skewnorm.pdf(x, a),
#        'r-', lw=5, alpha=0.6, label='skewnorm pdf')
# rv = skewnorm(a)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
# vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
# print("dd:", np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a)))
# print("a2:", a)
r = skewnorm.rvs(a, size=64)
r = 0.15 * r + 3.1
print("r :", r)
# ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.hist(r, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()