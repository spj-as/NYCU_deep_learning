import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(13)


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape((-1, 1))


def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50000)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bias", type=bool, default=True)

    return parser


def function(x, func):
    if func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    else:  # relu
        return np.maximum(0, x)


def der_function(z, func):
    if func == "sigmoid":
        return z * (1.0 - z)
    else:  # relu
        return (z > 0) * 1


def loss_function(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def gradient_descent(gradient, start, lr, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -lr * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.show()


def plot_curve(losses, epochs_num):
    epochs = np.array([idx for idx in range(epochs_num)])
    plt.plot(epochs, losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


class NNModel:
    def __init__(self, lr, hidden_size, bias_exist):
        self.x = None
        self.z = [
            np.zeros((hidden_size, 1)),
            np.zeros((hidden_size, 1)),
            np.zeros((1, 1)),
        ]
        self.weight = [
            np.random.randn(2, hidden_size),
            np.random.randn(hidden_size, hidden_size),
            np.random.randn(hidden_size, 1),
        ]
        self.bias = [
            np.full((hidden_size, 1), 0.01),
            np.full((hidden_size, 1), 0.01),
            np.full((1, 1), 0.01),
        ]
        self.lr = lr
        self.eps = 1e-5
        self.bias_exist = bias_exist

    def forward(self, x):
        # input
        self.x = x
        # h1
        self.z[0] = self.weight[0].T @ self.x
        if self.bias_exist:
            self.z[0] += self.bias[0]

        # h2
        self.z[1] = self.weight[1].T @ self.z[0]
        if self.bias_exist:
            self.z[1] += self.bias[1]

        # output
        self.z[2] = self.weight[2].T @ self.z[1]
        if self.bias_exist:
            self.z[2] += self.bias[2]

        return self.z[2]

    def backward(self, y, y_pred):
        # calculate the gradients
        input_size = y.shape[1]

        # output layer
        # dloss_dy_pred = - (y / (y_pred + self.eps)  - (1 - y) / (1 - y_pred +  self.eps))
        dloss_dy_pred = -2 * (y - y_pred) / y.shape[0]
        dloss_dweight2 = dloss_dy_pred @ self.z[1].T * (1 / input_size)
        if self.bias_exist:
            dloss_dbias2 = np.sum(dloss_dy_pred, axis=1, keepdims=True) * (
                1 / input_size
            )

        # second hidden layer
        dloss_dout2 = self.weight[2] @ dloss_dy_pred
        dloss_dweight1 = dloss_dout2 @ self.z[0].T * (1 / input_size)
        if self.bias_exist:
            dloss_dbias1 = np.sum(dloss_dout2, axis=1, keepdims=True) * (1 / input_size)

        # first hidden layer
        dloss_dout1 = self.weight[1] @ dloss_dout2
        dloss_dweight0 = dloss_dout1 @ self.x.T * (1 / input_size)
        if self.bias_exist:
            dloss_dbias0 = np.sum(dloss_dout1, axis=1, keepdims=True) * (1 / input_size)

        # update the weights
        self.weight[0] -= self.lr * dloss_dweight0.T
        self.weight[1] -= self.lr * dloss_dweight1.T
        self.weight[2] -= self.lr * dloss_dweight2.T

        if self.bias_exist:
            self.bias[0] -= self.lr * dloss_dbias0
            self.bias[1] -= self.lr * dloss_dbias1
            self.bias[2] -= self.lr * dloss_dbias2


def train(model, input, label, epochs):
    losses = []
    out = []
    for epoch in range(epochs):
        output = model.forward(input)
        # calculate the loss
        model.backward(label, output)
        losses.append(loss_function(label, output))
        if epoch % 2500 == 0:
            loss = loss_function(label, output)
            accuracy = ((label == np.round(output)).sum() / len(label[0])) * 100
            print("epoch %d loss : %f " % (epoch, loss))
            out.append([loss])

    print("accuracy =  %.0f %%" % accuracy)
    print(np.array(out).reshape(len(out), 1))
    show_result(input.T, label.T, output.T)
    plot_curve(losses, epochs)


def test(model, input, label):
    prediction = []
    for idx, x in enumerate(input.T):
        y_pred = model.forward(x.reshape(2, 1))
        loss = loss_function(label.T[idx][0], y_pred)
        print(
            "Iter%d |     Ground truth: %d |     prediction: %f |"
            % (idx, label[0][idx], y_pred)
        )
        prediction.append(y_pred.flatten())
        # print(y_pred.T)
    prediction = np.array(prediction)

    show_result(input.T, label.T, prediction)
    accuracy = ((label == np.round(prediction.T)).sum() / len(label[0])) * 100
    print("loss : %.10f accuracy =  %.2f %%" % (loss, accuracy))


if __name__ == "__main__":
    args = getParser()
    args = args.parse_args()
    epochs = args.epoch
    lr = args.lr
    hidden_size = args.hidden_size
    bias = args.bias

    # generate data
    x1, y1 = generate_linear(n=100)
    x2, y2 = generate_XOR_easy()
    test_x1, test_y1 = generate_linear(n=100)
    test_x2, test_y2 = generate_XOR_easy()
    # Data: Linear
    print("---------------------------------")
    print("| Data :", "Linear", "                |")
    print("| Epochs :", epochs, "              |")
    print("| Learning rate :", lr, "          |")
    print("| Hidden size :", hidden_size, "             |")
    print("| Bias :", bias, "                  |")
    print("---------------------------------")
    model = NNModel(
        lr=lr,
        hidden_size=hidden_size,
        bias_exist=bias,
    )
    print("------------train------------")
    train(model, x1.T, y1.T, epochs)
    print("------------test------------")
    test(model, test_x1.T, test_y1.T)

    # Data: XOR
    print("---------------------------------")
    print("| Data :", "XOR", "                   |")
    print("| Epochs :", epochs, "              |")
    print("| Learning rate :", lr, "          |")
    print("| Hidden size :", hidden_size, "             |")
    print("| Bias :", bias, "                  |")
    print("---------------------------------")
    model = NNModel(
        lr=lr,
        hidden_size=hidden_size,
        bias_exist=bias,
    )

    print("------------train------------")
    train(model, x2.T, y2.T, epochs)
    print("------------test------------")
    test(model, test_x2.T, test_y2.T)
