"""
The following code is not meant to be executed, but rather to show how a trade-off
between variance and bias was sought to be achieved.
The function collects the values, which are eventually plotted
"""

def bias_var(X, y):
    t_size = int(len(X)*0.66)
    X_train = X[:t_size]
    y_train = y[:t_size]
    X_test = X[t_size:]
    y_test = y[t_size:]
    
    preds = []
    # The loop is iterated over a couple of equivalent parameters
    # to reduce the execution time, at the expense of reduction of results 
    for leave, samp in enumerate(range(2, 101)):
        dt = DecisionTreeClassifier(max_leaf_nodes = leave+2, min_samples_split = samp)
        model = dt.fit(X, y)
        preds += [ list(model.predict(X)) ] # every element is an y_pred
       
    stats = []    
    for x in range(len(preds)):       
        dt_bias = (y - np.mean(preds[x]))**2
        dt_variance = np.var(preds[x])
        dt_error = (preds[x] - y)**2
        acc = accuracy_score(y_true = y, y_pred = preds[x])
        stats += [ (dt_bias.mean(), dt_variance, dt_error.mean(), round(acc, 6)) ]
       
    stats = [ list(x) for x in stats ]
    df = pd.DataFrame(stats, columns = ["error", "bias", "variance", "accuracy"])
    df["bias_plus_var"] = df.bias + df.variance
    
    return df


stats_df = bias_var(X_tree, y_tree)

print(stats_df.loc[stats_df.bias_plus_var == min(stats_df.bias_plus_var)])
print()
print(stats_df.loc[stats_df.accuracy == max(stats_df.accuracy)])

plt.style.use('fivethirtyeight')
fig, ax1 = plt.subplots(figsize = (9, 5))

ax1.set_xlabel('Model Complexity')
ax1.set_ylabel('Bias$^2$ + Variance', color = "#3DA5D9")
ax1.plot(range(len(stats_df)), stats_df.bias_plus_var, color = "#3DA5D9", linewidth = 1, alpha = 0.7)
ax1.plot(range(len(stats_df)), stats_df.variance, color = "#DB504A", linewidth = 1, alpha = 0.7)

ax1.tick_params(axis = 'y')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color = "#EA7317")
ax2.plot(range(len(stats_df)), stats_df.accuracy, color = "#EA7317", linewidth = 1, alpha = 0.7)
ax2.tick_params(axis = 'y')
fig.tight_layout()