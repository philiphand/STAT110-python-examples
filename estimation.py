import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Estimation of unknown quantities (prinsipper for estimering av ukjente størrelser)
# One of the main goals of statistics is to estimate unknown quantities based on sample data.
# Examples of these unknown quantities are population means, population variances, and proportions.
# To estimate these quantities, we often use sample statistics such as the sample mean, sample variance, and sample proportion.

# Let's generate a random sample from a normal distribution with a mean of 50 and a standard deviation of 10.
np.random.seed(42)
population_mean = 50
population_std = 10
sample_size = 100

sample = np.random.normal(loc=population_mean, scale=population_std, size=sample_size)

# We can use the sample mean as an estimate for the population mean
sample_mean = np.mean(sample)
print("Estimated population mean:", sample_mean)

# Confidence intervals (konfidensintervaller)
# Confidence intervals provide a range of values that likely contain the true population parameter.
# The most common confidence level is 95%, but other levels (e.g., 90% or 99%) can be used depending on the context.

# Calculate the 95% confidence interval for the population mean
confidence_level = 0.95
alpha = 1 - confidence_level
z_score = stats.norm.ppf(1 - alpha / 2)
margin_of_error = z_score * (population_std / np.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print("95% confidence interval for population mean:", confidence_interval)

# Hypothesis testing (hypotesetesting)
# Hypothesis testing is a method for testing a claim or hypothesis about a population parameter.
# The two main types of hypothesis tests are the null hypothesis (H0) and the alternative hypothesis (H1).

# Let's test if the population mean is different from a specific value (e.g., 55)
null_hypothesis_mean = 55
test_statistic = (sample_mean - null_hypothesis_mean) / (population_std / np.sqrt(sample_size))

# Calculate the p-value for a two-tailed test
p_value = 2 * (1 - stats.norm.cdf(np.abs(test_statistic)))
print("p-value:", p_value)

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("We reject the null hypothesis (H0) and accept the alternative hypothesis (H1).")
else:
    print("We fail to reject the null hypothesis (H0).")

# Plot the sample data, estimated mean, and confidence interval
plt.hist(sample, bins=20, alpha=0.5, color='blue', label='Sample Data')
plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(confidence_interval[0], color='green', linestyle='dashed', linewidth=2, label='95% Confidence Interval')
plt.axvline(confidence_interval[1], color='green', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Sample Data, Estimated Mean, and Confidence Interval')
plt.show()
