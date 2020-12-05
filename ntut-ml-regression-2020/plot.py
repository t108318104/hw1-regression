import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics
train = pd.read_csv('train-v3.csv')


a=np.array(['price','sale_yr','sale_month','sale_day','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
   'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode',
   'lat','long','sqft_living15','sqft_lot15'])


 
fig, ax = plt.subplots()
ax.scatter(x = train[a[0]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[0], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[1]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[1], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[2]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[2], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[3]], y = train['price'])
plt.xlabel(a[3], fontsize=13)
plt.ylabel('price', fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[4]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[4], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[5]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[5], fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train[a[6]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[6], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[7]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[7], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[8]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[8], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[9]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[9], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[10]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[10], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[11]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[11], fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train[a[12]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[12], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[13]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[13], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[14]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[14], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[15]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[15], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[16]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[16], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[17]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[17], fontsize=13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train[a[18]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[18], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[19]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[19], fontsize=13)
plt.show()

 
fig, ax = plt.subplots()
ax.scatter(x = train[a[20]], y = train['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel(a[20], fontsize=13)
plt.show()
