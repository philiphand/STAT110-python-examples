import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Concept 1: Simultanfordelinger (Joint Distributions)
# Simultanfordelinger, or joint distributions, describe the probability distribution of two or more random variables simultaneously.
# They allow us to understand the relationship between multiple variables.

# Example: Let's consider two random variables, X and Y, representing the age and income of a group of people.
ages = [20, 25, 30, 40, 50, 60, 70]
incomes = [30000, 40000, 50000, 60000, 70000, 80000, 90000]

# We can represent the joint distribution of X and Y using a 2D array.
joint_distribution = np.random.rand(len(ages), len(incomes))

# Concept 2: Marginal Distributions
# Marginal distributions represent the probability distribution of a single variable, ignoring the others.
# To obtain the marginal distribution, we sum the probabilities along the rows or columns of the joint distribution.

# Example: Compute the marginal distributions of X (age) and Y (income).
marginal_distribution_X = np.sum(joint_distribution, axis=1)
marginal_distribution_Y = np.sum(joint_distribution, axis=0)

# Concept 3: Conditional Distributions
# Conditional distributions represent the probability distribution of one variable, given the value of another variable.

# Example: Compute the conditional distribution of Y (income) given X (age) = 30.
conditional_distribution_Y_given_X = joint_distribution[2, :] / marginal_distribution_X[2]

# Concept 4: Covariance
# Covariance is a measure of how two variables change together. It indicates the direction of the relationship between the variables.
# If the covariance is positive, it means that as one variable increases, the other variable also increases, and vice versa.
# If the covariance is negative, it means that as one variable increases, the other variable decreases, and vice versa.

# Example: Compute the covariance between X (age) and Y (income).
data = {'X': ages, 'Y': incomes}
df = pd.DataFrame(data)
cov_XY = df.cov().iloc[0, 1]

# Concept 5: Correlation (Korrelasjon)
# Correlation is a measure of the strength and direction of the relationship between two variables.
# It is normalized, meaning it ranges from -1 to 1, where -1 indicates a strong negative relationship, 1 indicates a strong positive relationship, and 0 indicates no relationship.

# Example: Compute the correlation between X (age) and Y (income).
correlation_XY = df.corr().iloc[0, 1]

# Visualizing the data and correlation
sns.scatterplot(x='X', y='Y', data=df)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title(f'Correlation between Age and Income: {correlation_XY:.2f}')
plt.show()
