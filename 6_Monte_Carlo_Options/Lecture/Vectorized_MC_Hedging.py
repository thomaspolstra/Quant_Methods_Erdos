S = 100
K=100
T = 75/252
sigma = .3
r = 0
q = 0
N=3500
steps = 1
epsilon = 1


A = geo_paths(S, T, sigma, steps, N, r)

delta = MC_delta(S, T, sigma, N, K, epsilon, r)



#Parameters for MC sim
dt = T / steps
dW = np.sqrt(dt) * np.random.normal(size=(steps, N))
increments = (r - q - (sigma**2) / 2) * dt + sigma * dW
log_returns = np.cumsum(increments, axis=0)
st = S * np.exp(log_returns)
ST = np.insert(st, 0, S, axis=0)



call_values = [np.exp(-r*(T-TTE[i]))*np.maximum(ST[i] - K, 0) for i in range(len(ST))]
put_values = [np.exp(-r*(T-TTE[i]))*np.maximum(-ST[i] + K, 0) for i in range(len(ST))]

# Create empty arrays to store call and put deltas for each iteration
call_deltas = []
put_deltas = []

Deltas = [MC_delta_vectorized(np.exp(-r*(T-TTE[i]))*ST[i], TTE[i], sigma, N, K, dW, epsilon=1, r=r) for i in range(len(ST))]
call_deltas = [Deltas[i]['call_deltas'] for i in range(len(Deltas))]
put_deltas = [Deltas[i]['put_deltas'] for i in range(len(Deltas))]

X_call = np.array([-call_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (T - TTE[i + 1]))) for i in range(steps)])
X_put = np.array([-put_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (T - TTE[i+1]))) for i in range(steps)])
call_hedge_values = np.sum(X_call, axis=0)
put_hedge_values = np.sum(X_put, axis=0)

new_call_values = call_values[-1] + call_hedge_values
new_put_values = put_values[-1] + put_hedge_values

MC_call = np.mean(new_call_values)
MC_put = np.mean(new_put_values)

SE_call = standard_error(new_call_values, T)
SE_put = standard_error(new_put_values, T)
# Calculate final option values with hedging
#call_values = (call_prices * np.exp(-r * T) + call_hedge_values)
#put_values = (put_prices * np.exp(-r * T) + put_hedge_values)

MC_call,SE_call,bs('c',100,100,T,r,sigma)







###########################################


S = 100
K=100
T = 75/252
sigma = .3
r = 0
q = 0
N=3500
steps = 1
epsilon = 1

DT = T / steps
TTE = [T - DT * i for i in range(0, steps + 1)]
TTS = T - np.array(TTE)

#Parameters for MC sim
dt = T / steps
dW = np.sqrt(dt) * np.random.normal(size=(steps, N))
increments = (r - q - (sigma**2) / 2) * dt + sigma * dW
log_returns = np.cumsum(increments, axis=0)
st = S * np.exp(log_returns)
ST = np.insert(st, 0, S, axis=0)



call_values = [np.exp(-r*(T-TTE[i]))*np.maximum(ST[i] - K, 0) for i in range(len(ST))]
put_values = [np.exp(-r*(T-TTE[i]))*np.maximum(-ST[i] + K, 0) for i in range(len(ST))]

# Create empty arrays to store call and put deltas for each iteration
call_deltas = []
put_deltas = []

Deltas = [MC_delta_vectorized(np.exp(-r*(T-TTE[i]))*ST[i], TTE[i], sigma, N, K, dW, epsilon=1, r=r) for i in range(len(ST))]
call_deltas = [Deltas[i]['call_deltas'] for i in range(len(Deltas))]
put_deltas = [Deltas[i]['put_deltas'] for i in range(len(Deltas))]

X_call = np.array([-call_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (T - TTE[i + 1]))) for i in range(steps)])
X_put = np.array([-put_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (T - TTE[i+1]))) for i in range(steps)])
call_hedge_values = np.sum(X_call, axis=0)
put_hedge_values = np.sum(X_put, axis=0)

new_call_values = call_values[-1] + call_hedge_values
new_put_values = put_values[-1] + put_hedge_values

MC_call = np.mean(new_call_values)
MC_put = np.mean(new_put_values)

SE_call = standard_error(new_call_values, T)
SE_put = standard_error(new_put_values, T)
# Calculate final option values with hedging
#call_values = (call_prices * np.exp(-r * T) + call_hedge_values)
#put_values = (put_prices * np.exp(-r * T) + put_hedge_values)

MC_call,SE_call,bs('c',100,100,T,r,sigma)