import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('darkgrid')
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
import scipy.stats as stats
import time

def geo_paths(S, T, sigma, steps, N, r=0, q=0):
    """
    Parameters:
    S: Initial value of the asset/position
    T: Time in years
    r: Risk-free interest rate
    q: Continuous dividend yield
    sigma: Yearly volatility of the stock
    steps: Number of steps in a simulation
    N: Number of simulations
    strike_price: Strike price for options
    
    Output: Simulated geometric Brownian motion paths of assets/positions based on the inputs.
    """
    
    dt = T / steps
    dW = np.sqrt(dt) * np.random.normal(size=(steps, N))
    increments = (r - q - (sigma**2) / 2) * dt + sigma * dW
    log_returns = np.cumsum(increments, axis=0)
    ST = S * np.exp(log_returns)
    paths_with_strike = np.insert(ST, 0, S, axis=0)
    return paths_with_strike


def mc_Euro(S, T, sigma, steps, K, N, r=0, q=0):
    """
    Perform a Monte Carlo simulation to estimate call and put option prices.
    
    :param S: Initial stock price
    :param T: Time to maturity in years
    :param sigma: Volatility of the stock price
    :param steps: Number of time steps in the simulation
    :param K: Strike price of the option
    :param N: Number of simulation paths
    :param r: Risk-free interest rate earned on asset, e.g.
    :return: Estimated call and put option prices, along with standard errors
    
    This function calculates option prices using a Monte Carlo simulation approach. It generates
    N simulation paths for the stock price using the given parameters. The payoffs for call and put
    options are calculated based on the final stock prices of each path.
    
    The function also computes the standard error of the estimated option prices using the
    `standard_error` function. Additionally, it calculates the percentage of paths where the
    options are in-the-money and computes the mean prices of these in-the-money paths to estimate
    the final call and put option prices.
    
    The Black-Scholes formula is used to calculate the theoretical option prices (`call_bs` and `put_bs`)
    for comparison with the simulation results. The estimated option prices, standard errors, and
    theoretical prices are returned as the function's output.
    """
    
    #generate stock prices
    stock_prices = geo_paths(S, T, sigma, steps, N, r, q)
    
    # Calculate payoffs for call and put options
    call_payoffs = np.maximum(stock_prices[-1] - K, 0)
    put_payoffs = np.maximum(K - stock_prices[-1], 0)
    
    call_prices = call_payoffs * np.exp(-r * T)
    put_prices = put_payoffs * np.exp(-r * T)
    
    # Find standard errors
    SE_call = standard_error(call_payoffs, r, T)
    SE_put = standard_error(put_payoffs, r, T)
    
    # Compute estimated call and put prices
    estimated_call_price = np.mean(call_prices) 
    estimated_put_price = np.mean(put_prices)

    # Calculate Black-Scholes option prices
    call_bs = bs('c', S, K, T, r, sigma)
    put_bs = bs('p', S, K, T, r, sigma)
    
    return {
        'call_bs': call_bs,
        'put_bs': put_bs,
        'estimated_call_price': estimated_call_price,
        'estimated_put_price': estimated_put_price,
        'SE_call': SE_call,
        'SE_put': SE_put,
        'simulated_call_prices' : call_prices,
        'simulated_put_prices' : put_prices,
        'simulated_stock_paths' : stock_prices
    }


def mc_plot(S, T, sigma, steps, N, r=0, q=0):
    """
    Generate and visualize Monte Carlo simulated stock price paths.
    
    :param S: Initial stock price
    :param T: Time to maturity in years
    :param sigma: Volatility of the stock price
    :param steps: Number of time steps in the simulation
    :param N: Number of simulation paths
    :param r: Risk-free interest rate
    :param q: Dividend yield
    :return: None
    
    This function uses Monte Carlo simulation to generate and visualize stock price paths over time.
    It takes the initial stock price, strike price, time to maturity, risk-free interest rate, dividend yield,
    volatility, number of time steps, and the total number of simulation paths as inputs.
    
    The function calls the `geo_paths` function to generate the simulated stock price paths based on the given
    parameters. It then plots these simulated paths using matplotlib, with the x-axis representing trading days
    and the y-axis representing the corresponding stock prices.
    
    The title of the plot is set to "Simulated Stock Paths," and the plot is displayed using `plt.show()`.
    """
    stock_prices = geo_paths(S, T, sigma, steps, N, 0, q)
    
    # Plotting the simulated stock paths
    plt.plot(stock_prices)
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Paths", size=20)
    plt.show()


def per_in_money_paths(returns):
    """
    Calculate the percentage of in-the-money paths based on the provided returns.

    :param returns: A list or array of returns or payoffs from financial transactions
    :return: Percentage of in-the-money paths
    
    This function calculates the percentage of in-the-money paths within a given set of returns or payoffs.
    It takes the provided returns, converts them to a NumPy array, and filters out the positive returns
    (i.e., in-the-money scenarios). The function then computes the ratio of in-the-money paths to the total
    number of paths, providing a measure of how often positive returns occur within the dataset.

    This percentage is useful for evaluating the likelihood of achieving profitable outcomes based on historical
    or simulated data. It is commonly used in options trading to assess the effectiveness of strategies and
    potential profitability.
    """
    M = len(returns)
    returns = np.array(returns)
    returns_positive = returns[returns > 0]
    return len(returns_positive) / M

def standard_error(payoffs, T, r=0):
    """
    Calculate the standard error of an array of payoffs.

    Parameters:
    payoffs (array-like): An array of payoffs or returns.
    T (float): Time in years.
    r (float, optional): Risk-free interest rate. Default is 0.

    Returns:
    float: The calculated standard error of the payoffs.

    Explanation:
    1. Calculate the mean payoff after discounting by the risk-free interest rate.
    2. Determine the number of payoffs (N) in the array.
    3. Calculate the sample standard deviation (sigma) of the payoffs.
    4. Compute the standard error (SE) as sigma divided by the square root of N.
    5. Return the calculated standard error.

    Note: This function assumes that the payoffs are independent and identically distributed.
    """

    # Calculate the mean payoff after discounting by the risk-free interest rate
    payoff = np.mean(payoffs) * np.exp(-r * T)

    # Determine the number of payoffs (N) in the array
    N = len(payoffs)

    # Calculate the sample standard deviation (sigma) of the payoffs
    sigma = np.sqrt(np.sum((payoffs - payoff)**2) / (N - 1))

    # Compute the standard error (SE) as sigma divided by the square root of N
    SE = sigma / np.sqrt(N)

    # Return the calculated standard error
    return SE

import numpy as np

def mc_Euro_static_hedge(S, T, sigma, steps, K, N, r=0, q=0):
    """
    Calculate the Monte Carlo simulated prices and hedge prices for European options using a static hedging strategy.
    
    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        steps (int): Number of time steps in the simulation.
        K (float): Strike price of the option.
        N (int): Number of simulations.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.
        
    Returns:
        dict: A dictionary containing calculated values.
            'call_bs': Black-Scholes value of the call option.
            'put_bs': Black-Scholes value of the put option.
            'MC_call_hedge_price': Monte Carlo simulated price of the call option using a static hedge strategy.
            'MC_put_hedge_price': Monte Carlo simulated price of the put option using a static hedge strategy.
            'SE_call': Standard error of the call option price.
            'SE_put': Standard error of the put option price.
    """
    mc_sim = mc_Euro(S, T, sigma, steps, K, N, r, q)
    call_prices = mc_sim['simulated_call_prices']
    put_prices = mc_sim['simulated_put_prices']
    A = mc_sim['simulated_stock_paths'] 
    
    
    d_call = delta('c', S, K, T, r, sigma)
    d_put = delta('p', S, K, T, r, sigma)
    
    call_bs = bs('c', S, K, T, r, sigma)
    put_bs = bs('p', S, K, T, r, sigma)
    
    call_hedge_prices = (call_prices  - d_call * (A[-1] - S ) ) * np.exp(-r*T)
    put_hedge_prices = (put_prices  + d_put * (A[-1] - S)) * np.exp(-r*T)
    
    MC_call_price = np.mean(call_hedge_prices)
    MC_put_price = np.mean(put_hedge_prices) 
    
    SE_call = standard_error(call_hedge_prices, T)
    SE_put = standard_error(put_hedge_prices, T)
    
    return {
        'call_bs': call_bs,
        'put_bs': put_bs,
        'MC_call_hedge_price': MC_call_price,
        'MC_put_hedge_price': MC_put_price,
        'SE_call': SE_call,
        'SE_put': SE_put
    }


def mc_Euro_static_hedge(S, T, sigma, steps, K, N, r=0, q=0):
    """
    Calculate the Monte Carlo simulated prices and hedge prices for European options using a static hedging strategy.
    
    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        steps (int): Number of time steps in the simulation.
        K (float): Strike price of the option.
        N (int): Number of simulations.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.
        
    Returns:
        dict: A dictionary containing calculated values.
            'call_bs': Black-Scholes value of the call option.
            'put_bs': Black-Scholes value of the put option.
            'MC_call_hedge_price': Monte Carlo simulated price of the call option using a static hedge strategy.
            'MC_put_hedge_price': Monte Carlo simulated price of the put option using a static hedge strategy.
            'SE_call': Standard error of the call option price.
            'SE_put': Standard error of the put option price.
    """
    mc_sim = mc_Euro(S, T, sigma, steps, K, N, r, q)
    call_prices = mc_sim['simulated_call_prices']
    put_prices = mc_sim['simulated_put_prices']
    A = mc_sim['simulated_stock_paths'] * np.exp(-r*T)
    
    
    d_call = delta('c', S, K, T, r, sigma)
    d_put = delta('p', S, K, T, r, sigma)
    
    call_bs = bs('c', S, K, T, r, sigma)
    put_bs = bs('p', S, K, T, r, sigma)
    
    call_hedge_prices = call_prices  - d_call * (A[-1] - S ) 
    put_hedge_prices = put_prices  + d_put * (A[-1] - S)
    
    MC_call_price = np.mean(call_hedge_prices)
    MC_put_price = np.mean(put_hedge_prices) 
    
    SE_call = standard_error(call_hedge_prices, T)
    SE_put = standard_error(put_hedge_prices, T)
    
    return {
        'call_bs': call_bs,
        'put_bs': put_bs,
        'MC_call_hedge_price': MC_call_price,
        'MC_put_hedge_price': MC_put_price,
        'SE_call': SE_call,
        'SE_put': SE_put
    }

def mc_sim_dynamic(S, T, sigma, steps, K, N, r, q):
    """
    Calculate the Monte Carlo simulated prices and hedge values for European options using a dynamic hedging strategy.
    
    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        steps (int): Number of time steps in the simulation.
        K (float): Strike price of the option.
        N (int): Number of simulations.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        
    Returns:
        dict: A dictionary containing calculated values.
            'call_bs': Black-Scholes value of the call option.
            'put_bs': Black-Scholes value of the put option.
            'MC_call_hedge_price': Monte Carlo simulated price of the call option using a dynamic hedge strategy.
            'MC_put_hedge_price': Monte Carlo simulated price of the put option using a dynamic hedge strategy.
            'SE_call': Standard error of the call option price with dynamic hedging.
            'SE_put': Standard error of the put option price with dynamic hedging.
            'MC_no_hedge': Monte Carlo simulated price of the option without any hedging.
            'MC_no_hedge_SE': Standard error of the option price without any hedging.
    """
    # Simulate stock price paths
    A = geo_paths(S, T, sigma, steps, N, r, q)
    
    # Calculate option prices at maturity
    call_prices = np.maximum(A[-1] - K, 0)
    put_prices = np.maximum(-A[-1] + K, 0)
    
    # Initialize arrays to store deltas at each time step for each simulation
    Deltas_call = np.zeros((steps + 1, N))
    Deltas_put = np.zeros((steps + 1, N))
    
    # Calculate time interval between steps and time to expiration at each step
    DT = T / steps
    TTE = [T - DT * i for i in range(0, steps + 1)]
    
    # Calculate deltas for the option contract at each time step and for each simulation
    for i in range(steps):
        time_to_expire = TTE[i]
        Deltas_call[i] = [delta('c', A[i][j] * np.exp(-r * (T - TTE[i])), K, time_to_expire, r, sigma) for j in range(N)]
        Deltas_put[i] = [delta('p', A[i][j] * np.exp(-r * (T - TTE[i])), K, time_to_expire, r, sigma) for j in range(N)]
        
    # Calculate hedging corrections and values for the call and put options
    X_call = np.array([-Deltas_call[i] * ((A[i + 1] - A[i]) * np.exp(-r * (T - TTE[i + 1]))) for i in range(steps)])
    X_put = np.array([-Deltas_put[i] * ((A[i + 1] - A[i]) * np.exp(-r * (T - TTE[i]))) for i in range(steps)])
    call_hedge_values = np.sum(X_call, axis=0)
    put_hedge_values = np.sum(X_put, axis=0)
    
    # Calculate final option values with hedging
    call_values = (call_prices * np.exp(-r * T) + call_hedge_values)
    put_values = (put_prices * np.exp(-r * T) + put_hedge_values)
    
    # Calculate Monte Carlo simulated option value without hedging and its standard error
    MC_no_hedge = np.mean(call_prices * np.exp(-r * T))
    MC_no_hedge_SE = standard_error(call_prices * np.exp(-r * T), T)
    
    # Calculate standard errors for the call and put option values with dynamic hedging
    SE_call = standard_error(call_values, T, r)
    SE_put = standard_error(put_values, T, r)
    call_bs = bs('c', S, K, T, r, sigma)
    put_bs = bs('p', S, K, T, r, sigma)
    
    return {
        'call_bs': call_bs,
        'put_bs': put_bs,
        'MC_call_hedge_price': np.mean(call_values),
        'MC_put_hedge_price': np.mean(put_values),
        'SE_call': SE_call,
        'SE_put': SE_put,
        'MC_no_hedge': MC_no_hedge,
        'MC_no_hedge_SE': MC_no_hedge_SE,
        'call_hedged_values': call_values,
        'put_heged_values': put_values,
        'call_values_no_hedge': call_prices * np.exp(-r * T)
    }

def MC_delta(S, T, sigma, N, K, epsilon=1, r=0, q=0):
    """
    Estimate the deltas of a European call and put option using a central difference method based on Monte Carlo simulations.

    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        N (int): Number of simulations.
        K (float): Strike price of the option.
        epsilon (float, optional): Small perturbation factor for delta estimation. Default is 1.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.

    Returns:
        tuple: A tuple containing estimated deltas for the call and put options.
            delta_call (float): Estimated delta for the call option.
            delta_put (float): Estimated delta for the put option.
    """

    dW = np.sqrt(T) * np.random.normal(size=(1, N))
    increments = (r - q - (sigma**2) / 2) * T + sigma * dW
    log_returns = np.cumsum(increments, axis=0)
    ST = S * np.exp(log_returns)
    ST1 = (S - epsilon) * np.exp(log_returns)
    ST2 = (S + epsilon) * np.exp(log_returns)
    
    call_values = np.exp(-r*T)*np.maximum(ST[-1] - K, 0)
    call_values1 = np.exp(-r*T)*np.maximum(ST1[-1] - K, 0)
    call_values2 = np.exp(-r*T)*np.maximum(ST2[-1] - K, 0)
    
    put_values = np.exp(-r*T)*np.maximum(-ST[-1] + K, 0)
    put_values1 = np.exp(-r*T)*np.maximum(-ST1[-1] + K, 0)
    put_values2 = np.exp(-r*T)*np.maximum(-ST2[-1] + K, 0)
    
    delta_call =  np.mean([(call_values2 - call_values) / epsilon, (call_values - call_values1) / epsilon])
    delta_put =  np.mean([(put_values2 - put_values) / epsilon, (put_values - put_values1) / epsilon])
    
    all_call_deltas =  np.array([(call_values2 - call_values) / epsilon, (call_values - call_values1) / epsilon])
    all_put_deltas =  np.array([(put_values2 - put_values) / epsilon, (put_values - put_values1) / epsilon])
    
    
    

    # Calculate the sample standard deviation of deltas
    sigma_call = np.sqrt(np.sum((all_call_deltas - delta_call)**2) / (N - 1)) 
    sigma_put = np.sqrt(np.sum((all_put_deltas - delta_put)**2) / (N - 1)) 
    SE_call = sigma_call / np.sqrt(N)
    SE_put = sigma_put / np.sqrt(N)

    
    return {'delta_call':delta_call, 'delta_put': delta_put, 'all_call_deltas': all_call_deltas, 'SE_call': SE_call, 'SE_put': SE_put, 
            'all_put_deltas': all_put_deltas}


import numpy as np

def MC_delta_vectorized(S, T, sigma, N, K, dW, epsilon=1, r=0, q=0):
    """
    Monte Carlo Simulation for Delta Estimation - Vectorized Approach 

    This function performs a Monte Carlo simulation to estimate deltas of European call and put options for a given array of stock prices.
    
    Parameters:
    S (array-like): Array of stock prices.
    T (float): Time to expiration of the options.
    sigma (float): Volatility of the underlying stock.
    N (int): Number of simulation paths.
    K (float): Strike price of the options.
    epsilon (float, optional): Small increment for delta calculation. Default is 1.
    dW (numpy array):  Wiener process increment,  dW = np.sqrt(T) * np.random.normal(size=(1, N))
    r (float, optional): Risk-free interest rate. Default is 0.
    q (float, optional): Dividend yield. Default is 0.

    Returns:
    dict: A dictionary containing estimated call and put deltas for each stock price.
    """
    if type(dW) != type(np.array([])):
        return(f'{dW} needs to be a numpy array')
    
    if len(dW[0]) != N:
        return(f'length of {dW} is not {N}: Try dW = np.random.normal(size=(1, {N}))')
    
    increments = (r - q - (sigma**2) / 2) * T + sigma * dW
    log_returns = np.cumsum(increments, axis=0)

    stock_paths = {}
    call_values = {}
    put_values = {}
    call_deltas = {}
    put_deltas = {}

    S = np.array(S)
    S1 = np.array(S)-epsilon
    S2 = np.array(S)+epsilon

    paths = np.array([s * np.exp(log_returns) for s in S])
    sim_call_values = np.array([np.exp(-r * T) * np.maximum(paths[i][-1] - K, 0) for i in range(len(S))])
    sim_put_values = np.array([np.exp(-r * T) * np.maximum(-paths[i][-1] + K, 0) for i in range(len(S))])

    paths1 = np.array([(s-epsilon) * np.exp(log_returns) for s in S])
    sim_call_values1 = np.array([np.exp(-r * T) * np.maximum(paths1[i][-1] - K, 0) for i in range(len(S))])
    sim_put_values1 = np.array([np.exp(-r * T) * np.maximum(-paths1[i][-1] + K, 0) for i in range(len(S))])

    paths2 = np.array([(s+epsilon) * np.exp(log_returns) for s in S])
    sim_call_values2 = np.array([np.exp(-r * T) * np.maximum(paths2[i][-1] - K, 0) for i in range(len(S))])
    sim_put_values2 = np.array([np.exp(-r * T) * np.maximum(-paths2[i][-1] + K, 0) for i in range(len(S))])

    call_deltas = np.array([(np.mean(sim_call_values2[i] - sim_call_values[i]) + np.mean(sim_call_values[i] - sim_call_values1[i]))/(2*epsilon) for i in range(len(S))])
    put_deltas = np.array([(np.mean(sim_put_values2[i] - sim_put_values[i]) + np.mean(sim_put_values[i] - sim_put_values1[i]))/(2*epsilon) for i in range(len(S))])
    
    return {'call_deltas': call_deltas, 'put_deltas': put_deltas}


def mc_full_sim(S, T, sigma, K, N, epsilon=1, r=0, q=0):
    """
    Calculate the Monte Carlo simulated prices and hedge prices for European options using a static hedging strategy without black scholes formulas to find delta.
    
    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        steps (int): Number of time steps in the simulation.
        K (float): Strike price of the option.
        N (int): Number of simulations.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.
        
    Returns:
        dict: A dictionary containing calculated values.
            'call_bs': Black-Scholes value of the call option.
            'put_bs': Black-Scholes value of the put option.
            'MC_call_hedge_price': Monte Carlo simulated price of the call option using a static hedge strategy.
            'MC_put_hedge_price': Monte Carlo simulated price of the put option using a static hedge strategy.
            'SE_call': Standard error of the call option price.
            'SE_put': Standard error of the put option price.
    """
    mc_sim = mc_Euro(S, T, sigma, 1, K, N, r, q)
    call_prices = mc_sim['simulated_call_prices']
    put_prices = mc_sim['simulated_put_prices']
    A = mc_sim['simulated_stock_paths'] * np.exp(-r*T)
    
    D = MC_delta(S, T, sigma, N, K, epsilon, r, q)
    d_call =  D['delta_call']
    d_put = D['delta_put']
    

    
    call_hedge_prices = call_prices  - d_call * (A[-1] - S ) 
    put_hedge_prices = put_prices  + d_put * (A[-1] - S)
    
    MC_call_price = np.mean(call_hedge_prices)
    MC_put_price = np.mean(put_hedge_prices) 
    
    SE_call = standard_error(call_hedge_prices, T)
    SE_put = standard_error(put_hedge_prices, T)
    
    return {
        'MC_call': MC_call_price,
        'MC_put': MC_put_price,
        'SE_call': SE_call,
        'SE_put': SE_put
    }

def MC_accurate(S, T, sigma, steps, K, N, r=0, q=0, epsilon=1):
    """
    Estimate the prices of European call and put options using a more accurate Monte Carlo simulation method.
    
    This method involves simulating the stock price path and considering hedging strategies to calculate option prices.

    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        steps (int): Number of time steps for the simulation.
        K (float): Strike price of the option.
        N (int): Number of simulations.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.
        epsilon (float, optional): Small perturbation factor for delta estimation. Default is 1.

    Returns:
        dict: A dictionary containing estimated Monte Carlo prices and standard errors for the call and put options.
            MC_call (float): Estimated Monte Carlo price for the call option.
            MC_put (float): Estimated Monte Carlo price for the put option.
            SE_call (float): Standard error of the Monte Carlo estimate for the call option.
            SE_put (float): Standard error of the Monte Carlo estimate for the put option.
    """
    
    # Calculate time step and time to expiration for each step
    DT = T / steps
    TTE = [T - DT * i for i in range(0, steps + 1)]
    
    # Generate random increments and calculate log returns for stock price simulation
    dt = T / steps
    dW = np.sqrt(dt) * np.random.normal(size=(steps, N))
    increments = (r - q - (sigma**2) / 2) * dt + sigma * dW
    log_returns = np.cumsum(increments, axis=0)
    st = S * np.exp(log_returns)
    ST = np.insert(st, 0, S, axis=0)

    # Calculate average stock values and option values at each time step
    avg_stock_values = [np.exp(-r*(T-TTE[i]))*np.mean(ST[i]) for i in range(len(ST))]
    call_values = [np.exp(-r*(T-TTE[i]))*np.maximum(ST[i] - K, 0) for i in range(len(ST))]
    put_values = [np.exp(-r*(T-TTE[i]))*np.maximum(-ST[i] + K, 0) for i in range(len(ST))]

    # Calculate deltas using previously defined MC_delta function
    DELTAS = [MC_delta(avg_stock_values[i], TTE[i], sigma, 1000, K, epsilon, r) for i in range(len(ST))]
    call_deltas = [DELTAS[i]['delta_call'] for i in range(len(ST))]
    put_deltas = [DELTAS[i]['delta_put'] for i in range(len(ST))]

    # Calculate hedge values and new option values
    X_call = [-call_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (TTE[i + 1]))) for i in range(steps)]
    X_put = [-put_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (TTE[i + 1]))) for i in range(steps)]
    call_hedge_values = np.sum(X_call, axis=0)
    put_hedge_values = np.sum(X_put, axis=0)
    new_call_values = call_values[-1] + call_hedge_values
    new_put_values = put_values[-1] + put_hedge_values

    # Calculate Monte Carlo estimates and standard errors
    MC_call = np.mean(new_call_values)
    MC_put = np.mean(new_put_values)
    SE_call = standard_error(new_call_values, T)
    SE_put = standard_error(new_put_values, T)
    
    return {'MC_call': MC_call,
           'MC_put': MC_put,
           'SE_call': SE_call,
           'SE_put': SE_put}

