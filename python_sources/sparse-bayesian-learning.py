import numpy as np  # linear algebra
import pandas as pd
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.dtype(np.float64)


class MathUtils:
    @staticmethod
    def length_squared(m):
        return np.sum((np.square(m)))

    @staticmethod
    def mean(m):
        num_rows, num_cols = m.shape
        return np.sum(m) / (num_rows * num_cols)

    @staticmethod
    def variance(m):
        num_rows, num_cols = m.shape
        avg = MathUtils.mean(m)

        val = 0
        for r in range(num_rows):
            for c in range(num_cols):
                val += (m[r, c] - avg) ** 2

        if num_rows * num_cols <= 1:
            return val
        else:
            return val / (num_rows * num_cols - 1)

    @staticmethod
    def compute_mean_squared_distance(samples):
        rs = 0.
        length = len(samples)
        for i in range(length):
            j = i + 1
            while j < length:
                d = MathUtils.length_squared(samples[i] - samples[j])
                rs += d
                j += 1
        return rs / ((length - 1) * length / 2)


class NumpyUtils:

    @staticmethod
    def equals(a, b, eps):
        num_rows_a, num_cols_a = a.shape
        num_rows_b, num_cols_b = b.shape

        if num_rows_a != num_rows_b or num_cols_a != num_cols_b:
            return False

        for r in range(num_rows_a):
            for c in range(num_rows_b):
                if abs(a[r, c] - b[r, c]) > eps:
                    return False
        return True

    @staticmethod
    def set_all_elements(matrix, value):
        num_rows, num_cols = matrix.shape
        for r in range(num_rows):
            for c in range(num_cols):
                matrix[r, c] = value

    @staticmethod
    def set_column(matrix, column_num, column):
        matrix[:, column_num] = column.flatten()

    @staticmethod
    def get_column(matrix, column_num):
        m = matrix[:, int(column_num)].copy()
        return np.reshape(m, (m.shape[0], 1))

    @staticmethod
    def remove_column(matrix, column_num):
        return np.delete(matrix, column_num, 1)

    @staticmethod
    def remove_row(matrix, row_num):
        return np.delete(matrix, row_num, 0)

    @staticmethod
    def set_subm(matrix, sub_matrix):
        num_rows, num_cols = sub_matrix.shape
        for c in range(num_cols):
            for r in range(num_rows):
                matrix[r, c] = sub_matrix[r, c]


class RadialBasisKernel:
    def __init__(self, gamma=0.05) -> None:
        super().__init__()
        self.gamma = gamma

    def calc(self, a, b):
        d = np.dot((a - b).transpose(), (a - b))
        return np.exp(-self.gamma * d)


class DecisionFunction:
    def __init__(self, alpha, b, kernel_function, basis_vectors) -> None:
        super().__init__()
        self.alpha = alpha
        self.b = b
        self.kernel_function = kernel_function
        self.basis_vectors = basis_vectors

    def calc(self, x):
        temp = 0
        for i in range(len(self.alpha)):
            temp += self.alpha[i] * self.kernel_function.calc(x, self.basis_vectors[i])
        return temp - self.b


class RVM:
    kernel = None
    eps = None
    tau = 0.001

    def set_kernel(self, kernl):
        self.kernel = kernl

    def set_eps(self, eps):
        self.eps = eps

    @staticmethod
    def compute_initial_alpha(phi, t, var):
        temp = MathUtils.length_squared(phi)
        temp2 = np.dot(phi.transpose(), t).item()

        return temp / (temp2 * temp2 / temp + var)

    def get_kernel_column(self, idx, x, col):
        num_rows, num_cols = x.shape

        for r in range(num_rows):
            col[r, 0] = self.kernel.calc(x[idx], x[r]) + self.tau

    def pick_initial_vector(self, x, t):
        num_rows, num_cols = x.shape

        k_col = np.zeros((num_rows, 1))

        max_projection = np.NINF
        max_idx = 0

        for r in range(num_rows):
            self.get_kernel_column(r, x, k_col)
            temp = np.dot(k_col.transpose(), t).item()
            temp = temp * temp / MathUtils.length_squared(k_col)

            if temp > max_projection:
                max_projection = temp
                max_idx = r

        return max_idx

    @staticmethod
    def find_next_best_alpha_to_update(S, Q, alpha, active_bases, search_all_alphas, eps):
        selected_idx = -1
        greatest_improvement = -1
        for i in range(len(S)):
            value = -1
            if active_bases[i] >= 0:
                idx = int(active_bases[i].item())
                s = alpha[idx] * S[i] / (alpha[idx] - S[i])
                q = alpha[idx] * Q[i] / (alpha[idx] - S[i])

                if q * q - s > 0:
                    if search_all_alphas is False:
                        new_alpha = s * s / (q * q - s)
                        cur_alpha = alpha[idx]
                        new_alpha = 1 / new_alpha
                        cur_alpha = 1 / cur_alpha

                        value = Q[i] * Q[i] / (S[i] + 1 / (new_alpha - cur_alpha)) - \
                                np.math.log(1 + S[i] * (new_alpha - cur_alpha))
                elif search_all_alphas and idx + 2 < len(alpha):
                    value = Q[i] * Q[i] / (S[i] - alpha[idx]) - np.math.log(1 - S[i] / alpha[idx])
            elif search_all_alphas:
                s = S[i]
                q = Q[i]

                if q * q - s > 0:
                    value = (Q[i] * Q[i] - S[i]) / S[i] + np.math.log(S[i] / (Q[i] * Q[i]))

            if value > greatest_improvement:
                greatest_improvement = value
                selected_idx = i

        if greatest_improvement > eps:
            return selected_idx
        else:
            return -1

    def do_train(self, x, t):
        num_rows, num_cols = x.shape

        active_bases = np.zeros((num_rows, 1))
        phi = np.zeros((num_rows, 1))
        alpha = np.zeros((1, 1))
        prev_aplha = np.zeros((1, 1))

        weight = np.zeros((1, 1))
        prev_weight = np.zeros((1, 1))

        tempv = np.array([])
        K_col = np.zeros((len(x), 1))
        var = MathUtils.variance(t) * 0.1

        NumpyUtils.set_all_elements(active_bases, -1)
        first_basis = self.pick_initial_vector(x, t)
        self.get_kernel_column(first_basis, x, K_col)
        active_bases[first_basis] = 0
        NumpyUtils.set_column(phi, 0, K_col)
        alpha[0] = self.compute_initial_alpha(phi, t, var)
        weight[0] = 1

        Q = np.zeros((num_rows, 1))
        S = np.zeros((num_rows, 1))
        sigma = np.array([])

        tempv2 = np.array([])
        tempv3 = np.array([])
        tempm = np.array([])

        search_all_alphas = False
        ticker = 0
        rounds_of_narrow_search = 100

        while True:
            sigma = np.dot(phi.transpose(), phi) / var
            for r in range(len(alpha)):
                sigma[r, r] += alpha[r]
            sigma = inv(sigma)
            weight = np.dot(sigma, np.dot(phi.transpose(), t)) / var

            if ticker == rounds_of_narrow_search:
                if NumpyUtils.equals(prev_aplha, alpha, self.eps) and NumpyUtils.equals(prev_weight, weight, self.eps):
                    break

                prev_aplha = alpha
                prev_weight = weight
                search_all_alphas = True
                ticker = 0

            else:
                search_all_alphas = False

            ticker += 1

            tempv = np.dot(phi, np.dot(sigma, np.dot(phi.transpose(), t) / var))

            for i in range(len(S)):
                if not search_all_alphas and active_bases[i] == -1:
                    continue

                if active_bases[i] != -1:
                    K_col = NumpyUtils.get_column(phi, active_bases[i].item())
                else:
                    self.get_kernel_column(i, x, K_col)

                tempv2 = K_col.transpose() / var
                tempv3 = np.dot(tempv2, phi)
                S[i] = np.dot(tempv2, K_col) - np.dot(tempv3, np.dot(sigma, tempv3.transpose()))
                Q[i] = np.dot(tempv2, t) - np.dot(tempv2, tempv)

            selected_idx = self.find_next_best_alpha_to_update(S, Q, alpha, active_bases, search_all_alphas, self.eps)

            if selected_idx == -1:
                if not search_all_alphas:
                    ticker = rounds_of_narrow_search
                    continue
                else:
                    break

            var = MathUtils.length_squared(t - np.dot(phi, weight)) / (
                    len(x) - len(weight) + np.dot(alpha.transpose(), sigma.diagonal()))

            if active_bases[selected_idx] >= 0:
                idx = int(active_bases[selected_idx].item())
                s = alpha[idx] * S[selected_idx] / (alpha[idx] - S[selected_idx])
                q = alpha[idx] * Q[selected_idx] / (alpha[idx] - S[selected_idx])

                if q * q - s > 0:
                    alpha[idx] = s * s / (q * q - s)
                else:
                    active_bases[selected_idx] = -1
                    phi = NumpyUtils.remove_column(phi, idx)
                    weight = NumpyUtils.remove_row(weight, idx)
                    alpha = NumpyUtils.remove_row(alpha, idx)

                    for i in range(len(active_bases)):
                        if active_bases[i] > idx:
                            active_bases[i] -= 1
            else:
                s = S[selected_idx]
                q = Q[selected_idx]

                if q * q - s > 0:
                    active_bases[selected_idx] = phi.shape[1]

                    # update alpha
                    tempv = np.zeros((alpha.shape[0] + 1, 1))
                    NumpyUtils.set_subm(tempv, alpha)
                    tempv[phi.shape[1]] = s * s / (q * q - s)
                    tempv, alpha = alpha, tempv

                    # update weight
                    tempv = np.zeros((weight.shape[0] + 1, 1))
                    NumpyUtils.set_subm(tempv, weight)
                    tempv[phi.shape[1]] = 0
                    tempv, weight = weight, tempv

                    # update phi
                    tempm = np.zeros((phi.shape[0], phi.shape[1] + 1))
                    NumpyUtils.set_subm(tempm, phi)
                    self.get_kernel_column(selected_idx, x, K_col)
                    NumpyUtils.set_column(tempm, phi.shape[1], K_col)
                    tempm, phi = phi, tempm

        dictionary = np.array([])
        final_weights = np.array([])

        for i in range(len(active_bases)):
            if active_bases[i] >= 0:
                dictionary = np.append(dictionary, x[i])
                final_weights = np.append(final_weights, weight[int(active_bases[i])])

        return DecisionFunction(
            final_weights,
            -np.sum(final_weights) * self.tau,
            self.kernel,
            dictionary
        )

    def train(self, x, t):
        assert x is not None and t is not None and t.shape[1] == 1 and \
               len(x) == len(t) and len(x) > 0, 'decision function trainer::train(x, t)' \
                                                'invalid inputs were given to this function\n' \
                                                'x.nr(): %d\nt.nr(): %d\nx.nc(): %d\nt.nc(): %d' % (
                                                    x.shape[0], t.shape[0], x.shape[1], t.shape[1])
        return self.do_train(
            x,
            t
        )


def predict(model, x_test):
    y = np.zeros((x_test.shape[0], 1))
    for i in range(x_test.shape[0]):
        y[i] = model.calc(x_test[i])
    return y


def get_mse(y, y_pred):
    return (np.square(y - y_pred)).mean(axis=0)


class TestTrainer:
    @staticmethod
    def sinc(x):
        if x == 0:
            return 1
        return np.math.sin(x) / x

    def run(self):
        samples = np.zeros((15, 1))
        labels = np.zeros((15, 1))

        i = -10
        while i <= 4:
            samples[10 + i] = i
            labels[10 + i] = self.sinc(i)
            i = i + 1

        gamma = 2. / MathUtils.compute_mean_squared_distance(samples)

        trainer = RVM()
        trainer.set_kernel(RadialBasisKernel(gamma))
        trainer.set_eps(0.001)

        predictor = trainer.train(samples, labels)

        print('value = %f,\t sinx(value) = %f,\t pred(value) = %f' % (2.5, self.sinc(2.5), predictor.calc(2.5)))
        print('value = %f,\t sinx(value) = %f,\t pred(value) = %f' % (0.1, self.sinc(0.1), predictor.calc(0.1)))
        print('value = %f,\t sinx(value) = %f,\t pred(value) = %f' % (-4, self.sinc(-4), predictor.calc(-4)))
        print('value = %f,\t sinx(value) = %f,\t pred(value) = %f' % (5.0, self.sinc(5.0), predictor.calc(5.0)))

        y_pred = predict(predictor, samples)
        mse = (np.square(y_pred - labels)).mean(axis=0)
        print("mse = %f" % mse)


class DataTrainer:

    @staticmethod
    def run():
        # Load the data into a pandas dataframe
        df = pd.read_csv('../input/Concrete_Data_Yeh.csv')

        # Rename the columns
        df.columns = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age', 'strength']

        # Extract the X and y data from the DataFrame
        X = df.drop('strength', axis=1)
        y = df['strength']

        # Create the train and test data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        gamma = 2. / MathUtils.compute_mean_squared_distance(x_train.values)

        trainer = RVM()
        trainer.set_kernel(RadialBasisKernel(gamma))
        trainer.set_eps(0.001)

        predictor = trainer.train(x_train.values, np.reshape(y_train.values, (y_train.values.shape[0], 1)))

        y_predict = predict(predictor, x_test.values)
        mse = get_mse(np.reshape(y_test.values, (y_test.values.shape[0], 1)), y_predict)
        print("mse = %f" % mse)


#TestTrainer().run()
DataTrainer().run()
