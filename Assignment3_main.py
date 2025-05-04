#importing required libraries for the assignment
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("Dataset.xlsx")
print(data.head())

print(data.describe())

input_features = data.drop(["Positive?"],axis =1)
print(input_features)
input_matrix = np.array(input_features.values)
print(input_matrix)

target_output = data["Positive?"]
print(target_output)
output_matrix = np.array(target_output.values)
print(output_matrix)

print('Sigmoid Function')

weight_hidden1 = np.random.rand(4,8)
print(weight_hidden1)

weight_hidden2 = np.random.rand(8,8)
print(weight_hidden2)

weight_output = np.random.rand(8,1)
print(weight_output)

#defining learning rate
lr = 0.05
# Sigmoid function :
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))

for epoch in range(200000):
    input_hidden1 = np.dot(input_matrix,weight_hidden1)
    output_hidden1 = sigmoid(input_hidden1)
    
    input_hidden2 = np.dot(output_hidden1,weight_hidden2)
    output_hidden2 = sigmoid(input_hidden2)
    
    input_output = np.dot(output_hidden2, weight_output)
    output_final = sigmoid(input_output)
    
    output_final = np.array(output_final).reshape(-1, 1)
    target_output = np.array(target_output).reshape(-1, 1)
    
    error = ((1 / 2) * (np.power((output_final - target_output), 2)))

    # ========== BACKPROPAGATION ========== #

    # Output Layer
    derror_dout = output_final - target_output
    dout_din = sigmoid_der(input_output)
    derror_dwo = np.dot(output_hidden2.T, derror_dout * dout_din)

    # Hidden Layer 2
    derror_hidden2 = np.dot(derror_dout * dout_din, weight_output.T)
    dhidden2_din = sigmoid_der(input_hidden2)
    derror_wh2 = np.dot(output_hidden1.T, derror_hidden2 * dhidden2_din)

    # Hidden Layer 1
    derror_hidden1 = np.dot(derror_hidden2 * dhidden2_din, weight_hidden2.T)
    dhidden1_din = sigmoid_der(input_hidden1)
    derror_wh1 = np.dot(input_features.T, derror_hidden1 * dhidden1_din)

    # Update weights
    weight_output -= lr * derror_dwo
    weight_hidden2 -= lr * derror_wh2
    weight_hidden1 -= lr * derror_wh1

# Final output
print("Error of final output")
print(error.tolist())
print("*"*100)
print("Weight for the first hidden layer")
print(weight_hidden1)
print("*"*100)
print("Weight for the second hidden layer")
print(weight_hidden2)
print("*"*100)
print("Weight for the output layer")
print(weight_output)
print("*"*100)
print("Final output after training:")
print(output_final.tolist())

#For accuracy
def accuracy_test():
    predictions = (output_final > 0.5).astype(int)

    # Compare with actual output
    correct = np.sum(predictions == target_output)
    total = target_output.shape[0]
    accuracy = (correct / total) * 100

    print("*" * 100)
    print(f"Accuracy of the model: {accuracy:.2f}%")
accuracy_test()

prediction_data = np.array([[1,0,0,1],[0,0,1,0],[1,0,1,0]])
hidden1_input = np.dot(prediction_data,weight_hidden1)
hidden1_output= sigmoid(hidden1_input)
hidden2_input = np.dot(hidden1_output,weight_hidden2)
hidden2_output = sigmoid(hidden2_input)
prediction_data_input = np.dot(hidden2_output,weight_output)
predicted_output = sigmoid(prediction_data_input)
print(predicted_output)


##Problem 2

print('Hyperbolic Tangent')

# Hyperbolic Tangent (htan) Activation Function
def htan(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
# htan derivative
def der_htan(x):
    return 1 - htan(x) * htan(x)

#weights
weight_hidden1 = np.random.rand(4,8)
weight_hidden2 = np.random.rand(8,8)
weight_output = np.random.rand(8,1)
# Training loop
for epoch in range(200000):
    # ======== FORWARD PASS ========
    input_hidden1 = np.dot(input_matrix, weight_hidden1)
    output_hidden1 = htan(input_hidden1)

    input_hidden2 = np.dot(output_hidden1, weight_hidden2)
    output_hidden2 = htan(input_hidden2)

    input_output = np.dot(output_hidden2, weight_output)
    output_final = htan(input_output)

    # Reshape outputs to match target
    output_final = np.array(output_final).reshape(-1, 1)
    target_output = np.array(target_output).reshape(-1, 1)

    # Loss (Mean Squared Error)
    error = ((1 / 2) * (np.power((output_final - target_output), 2)))

    # ======== BACKPROPAGATION ========

    # Output Layer
    derror_dout = output_final - target_output
    dout_din = der_htan(input_output)
    derror_dwo = np.dot(output_hidden2.T, derror_dout * dout_din)

    # Hidden Layer 2
    derror_hidden2 = np.dot(derror_dout * dout_din, weight_output.T)
    dhidden2_din = der_htan(input_hidden2)
    derror_wh2 = np.dot(output_hidden1.T, derror_hidden2 * dhidden2_din)

    # Hidden Layer 1
    derror_hidden1 = np.dot(derror_hidden2 * dhidden2_din, weight_hidden2.T)
    dhidden1_din = der_htan(input_hidden1)
    derror_wh1 = np.dot(input_matrix.T, derror_hidden1 * dhidden1_din)

    # Update weights
    weight_output -= lr * derror_dwo
    weight_hidden2 -= lr * derror_wh2
    weight_hidden1 -= lr * derror_wh1

# Final output after training
print("Error of final output")
print(error.tolist())
print("*"*100)
print("Weight for the first hidden layer")
print(weight_hidden1)
print("*"*100)
print("Weight for the second hidden layer")
print(weight_hidden2)
print("*"*100)
print("Weight for the output layer")
print(weight_output)
print("*"*100)
print("Final output after training:")
print(output_final.tolist())

#For accuracy
accuracy_test()

prediction_data = np.array([[1,0,0,1],[0,0,1,0],[1,0,1,0]])
hidden1_input = np.dot(prediction_data,weight_hidden1)
hidden1_output= htan(hidden1_input)
hidden2_input = np.dot(hidden1_output,weight_hidden2)
hidden2_output = htan(hidden2_input)
prediction_data_input = np.dot(hidden2_output,weight_output)
predicted_output = htan(prediction_data_input)
print(predicted_output)

print('ReLU')

# Rectified Linear Unit (ReLU)
def ReLU(x):
    shape = x.shape
    data = [max(0, value) for value in np.array(x).flatten()]
    return np.array(data, dtype=float).reshape(shape)

# Derivative for ReLU
def der_ReLU(x):
    shape = x.shape
    data = [1 if value > 0 else 0 for value in np.array(x).flatten()]
    return np.array(data, dtype=float).reshape(shape)
    
# Training loop
lr = 0.001
weight_hidden1 = np.random.rand(4,8)
weight_hidden2 = np.random.rand(8,8)
weight_output = np.random.rand(8,1)
for epoch in range(200000):
    # Forward Pass
    input_hidden1 = np.dot(input_matrix, weight_hidden1)
    output_hidden1 = ReLU(input_hidden1)

    input_hidden2 = np.dot(output_hidden1, weight_hidden2)
    output_hidden2 = ReLU(input_hidden2)

    input_output = np.dot(output_hidden2, weight_output)
    output_final = ReLU(input_output)

    # Reshape outputs
    output_final = output_final.reshape(-1, 1)
    target_output = np.array(target_output).reshape(-1, 1)

    error = ((1 / 2) * (np.power((output_final - target_output), 2)))

    # ========== BACKPROPAGATION ========== #

    # Output Layer
    derror_dout = output_final - target_output
    dout_din = der_ReLU(input_output)
    derror_dwo = np.dot(output_hidden2.T, derror_dout * dout_din)

    # Hidden Layer 2
    derror_hidden2 = np.dot(derror_dout * dout_din, weight_output.T)
    dhidden2_din = der_ReLU(input_hidden2)
    derror_wh2 = np.dot(output_hidden1.T, derror_hidden2 * dhidden2_din)

    # Hidden Layer 1
    derror_hidden1 = np.dot(derror_hidden2 * dhidden2_din, weight_hidden2.T)
    dhidden1_din = der_ReLU(input_hidden1)
    derror_wh1 = np.dot(input_matrix.T, derror_hidden1 * dhidden1_din)

    # Update weights
    weight_output -= lr * derror_dwo
    weight_hidden2 -= lr * derror_wh2
    weight_hidden1 -= lr * derror_wh1

# Final output after training
print("Error of final output")
print(error.tolist())
print("*"*100)
print("Weight for the first hidden layer")
print(weight_hidden1)
print("*"*100)
print("Weight for the second hidden layer")
print(weight_hidden2)
print("*"*100)
print("Weight for the output layer")
print(weight_output)
print("*"*100)
print("Final output after training:")
print(output_final.tolist())

#For accuracy
accuracy_test()

prediction_data = np.array([[1,0,0,1],[0,0,1,0],[1,0,1,0]])
hidden1_input = np.dot(prediction_data,weight_hidden1)
hidden1_output= ReLU(hidden1_input)
hidden2_input = np.dot(hidden1_output,weight_hidden2)
hidden2_output = ReLU(hidden2_input)
prediction_data_input = np.dot(hidden2_output,weight_output)
predicted_output = ReLU(prediction_data_input)
print(predicted_output)

print('Leaky ReLU')

def leaky_ReLU(x):
    shape = x.shape
    x_flat = x.flatten()
    data = [max(0.05 * value, value) for value in x_flat]
    return np.array(data, dtype=float).reshape(shape)

def der_leaky_ReLU(x):
    shape = x.shape
    x_flat = x.flatten()
    data = [1 if value > 0 else 0.05 for value in x_flat]
    return np.array(data, dtype=float).reshape(shape)

# Training loop
lr = 0.001
weight_hidden1 = np.random.rand(4,8)
weight_hidden2 = np.random.rand(8,8)
weight_output = np.random.rand(8,1)

def clip_gradients(grad, clip_value=1.0):
    return np.clip(grad, -clip_value, clip_value)

for epoch in range(200000):
    # Forward Pass
    input_hidden1 = np.dot(input_matrix, weight_hidden1)
    output_hidden1 = leaky_ReLU(input_hidden1)

    input_hidden2 = np.dot(output_hidden1, weight_hidden2)
    output_hidden2 = leaky_ReLU(input_hidden2)

    input_output = np.dot(output_hidden2, weight_output)
    output_final = leaky_ReLU(input_output)

    # Reshape outputs
    output_final = output_final.reshape(-1, 1)
    target_output = np.array(target_output).reshape(-1, 1)

    error = ((1 / 2) * (np.power((output_final - target_output), 2)))

    # ========== BACKPROPAGATION ========== #

    # Output Layer
    derror_dout = output_final - target_output
    dout_din = der_leaky_ReLU(input_output)
    derror_dwo = np.dot(output_hidden2.T, derror_dout * dout_din)
    derror_dwo = clip_gradients(derror_dwo)

    # Hidden Layer 2
    derror_hidden2 = np.dot(derror_dout * dout_din, weight_output.T)
    dhidden2_din = der_leaky_ReLU(input_hidden2)
    derror_wh2 = np.dot(output_hidden1.T, derror_hidden2 * dhidden2_din)

    # Hidden Layer 1
    derror_hidden1 = np.dot(derror_hidden2 * dhidden2_din, weight_hidden2.T)
    dhidden1_din = der_leaky_ReLU(input_hidden1)
    derror_wh1 = np.dot(input_matrix.T, derror_hidden1 * dhidden1_din)

    # Update weights
    weight_output -= lr * derror_dwo
    weight_hidden2 -= lr * derror_wh2
    weight_hidden1 -= lr * derror_wh1

# Final output after training
print("Error of final output")
print(error.tolist())
print("*"*100)
print("Weight for the first hidden layer")
print(weight_hidden1)
print("*"*100)
print("Weight for the second hidden layer")
print(weight_hidden2)
print("*"*100)
print("Weight for the output layer")
print(weight_output)
print("*"*100)
print("Final output after training:")
print(output_final.tolist())

#For accuracy
predictions = (output_final > 0.5).astype(int)

# Compare with actual output
correct = np.sum(predictions == target_output)
total = target_output.shape[0]
accuracy = (correct / total) * 100

print("*" * 100)
print(f"Accuracy of the model: {accuracy:.2f}%")

prediction_data = np.array([[1,0,0,1],[0,0,1,0],[1,0,1,0]])
hidden1_input = np.dot(prediction_data,weight_hidden1)
hidden1_output= leaky_ReLU(hidden1_input)
hidden2_input = np.dot(hidden1_output,weight_hidden2)
hidden2_output = leaky_ReLU(hidden2_input)
prediction_data_input = np.dot(hidden2_output,weight_output)
predicted_output = leaky_ReLU(prediction_data_input)
print(predicted_output)