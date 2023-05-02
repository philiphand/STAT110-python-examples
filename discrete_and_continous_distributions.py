import numpy as np

# 1. Discrete and Continuous Distributions
# Discrete distributions are those where the variable can only take specific values (e.g., integers).
# Continuous distributions are those where the variable can take any value within a specified range.

# 2. Binomial Distribution
# It describes the number of successes in a fixed number of Bernoulli trials with the same probability of success.
# Example: Tossing a fair coin (n=10) with a probability of getting heads (p=0.5) in each toss.
n = 10
p = 0.5
binomial_sample = np.random.binomial(n, p, 1000) # 1000 samples from the binomial distribution

# 3. Hypergeometric Distribution
# It describes the probability of getting a certain number of successes in a sample without replacement.
# Example: Selecting 10 cards (n=10) from a deck of 52 cards with 26 red cards (M=26) and 26 black cards (N-M=26).
M = 26
N = 52
n = 10
hypergeometric_sample = np.random.hypergeometric(M, N-M, n, 1000) # 1000 samples from the hypergeometric distribution

# 4. Exponential Distribution
# It describes the time between events in a Poisson process with a constant rate.
# Example: Time between incoming calls at a call center with an average rate of 2 calls per hour (lam=2).
lam = 2
exponential_sample = np.random.exponential(1/lam, 1000) # 1000 samples from the exponential distribution

# 5. Poisson Distribution
# It describes the number of events occurring in a fixed interval of time, given a constant rate.
# Example: Number of emails received in a day with an average rate of 10 emails per day (lam=10).
lam = 10
poisson_sample = np.random.poisson(lam, 1000) # 1000 samples from the Poisson distribution

# 6. Normal Distribution (Gaussian Distribution)
# It is a continuous probability distribution, and it's characterized by a bell-shaped curve.
# Example: Heights of adult males with a mean height of 175 cm (mu=175) and a standard deviation of 7 cm (sigma=7).
mu = 175
sigma = 7
normal_sample = np.random.normal(mu, sigma, 1000) # 1000 samples from the normal distribution
