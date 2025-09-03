import pandas as pd  # Import pandas library for data manipulation

def simple_model(w1, w0, x):
    # Simple linear model: y = w1 * x + w0
    return w1 * x + w0

def loss(y_true, y_pred):
    # Calculate squared error between true and predicted values
    return (y_true - y_pred) ** 2

def cost_function(y_true, y_pred):
    # Calculate mean squared error (average loss)
    return loss(y_true, y_pred).mean()

def return_best_model(data):
    # Initialize best_cost to positive infinity, so any real cost will be lower
    best_cost = float('inf')
    best_w1 = None  # To store the best w1 value
    best_w0 = None  # To store the best w0 value

    # Loop through all possible values of w1
    for i in weight_dic['w1']:
        # Loop through all possible values of w0
        for j in weight_dic['w0']:
            # Calculate predictions using the current w1 and w0
            data['y_pred_w1_{}_w0_{}'.format(i, j)] = simple_model(i, j, data['x'])
            # Compute the cost for these predictions
            current_cost = cost_function(data['y'], data['y_pred_w1_{}_w0_{}'.format(i, j)])
            # If the current cost is lower
            if current_cost < best_cost:
                best_cost = current_cost
                best_w1 = i
                best_w0 = j

    return best_w1, best_w0, best_cost

if __name__ == "__main__":
    # Create a DataFrame with columns 'x' and 'y' and sample data
    data = pd.DataFrame(columns=('x', 'y'), data=[(1, 100), (2, 200), (3, 300)])
    # Dictionary containing possible values for weights w1 and w0
    weight_dic = {'w1': [100, 0.5], 'w0': [0, 10]}
    
    best_w1, best_w0, best_cost = return_best_model(data)
    print("Best w1:", best_w1)
    print("Best w0:", best_w0)
    print("Best cost:", best_cost)