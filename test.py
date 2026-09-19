import numpy as np
hours= np.array([1, 2, 3, 4, 5],dtype=float)
marks =np.array([20,30,40,50,60],dtype=float)


weight =0.0
bias=0.0

learning_rate=0.01

epochs=1000000

for epoch in range(epochs):

    prediction = weight * hours + bias

    error = prediction - marks

    weight_gradient = np.mean(error * hours)
    bias_gradient = np.mean(error)

    weight = weight - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient

    if epoch % 100 == 0:
        loss = np.mean(error ** 2)

        print(
            "Epoch:", epoch,
            "Weight:", weight,
            "Bias:", bias,
            "Loss:", loss
        )