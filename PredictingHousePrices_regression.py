from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

file = open("dataset.txt")

x = []
y = []
temp = []

for _line in file:
   _line = _line.replace('\n', '')
   temp = _line.split(',')
   y.append(int(temp[0]))
   x.append([int(temp[1]),int(temp[2])])

xnp = np.array(x)


# Split the data into training/testing sets
diabetes_X_train = xnp[:-12]
diabetes_X_test = xnp[-12:]

# Split the targets into training/testing sets
diabetes_y_train = y[:-12]
diabetes_y_test = y[-12:]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

print(diabetes_y_test)
print(regr.predict(diabetes_X_test))



diabetes_X0 = diabetes_X_test[:,0]
diabetes_X1 = diabetes_X_test[:,1]

# Plot outputs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(diabetes_X0,diabetes_X1, diabetes_y_test,  color='black')
ax.plot(diabetes_X0, diabetes_X1,regr.predict(diabetes_X_test), color='blue',linewidth=3)
#plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
#plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',linewidth=3)

#plt.xticks(())
#plt.yticks(())

ax.set_xlabel('bedrooms')
ax.set_ylabel('Size')
ax.set_zlabel('price')

plt.show()

