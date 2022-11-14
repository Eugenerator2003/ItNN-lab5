# from statistics import mean
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import uniform
import os
from PIL import Image, ImageOps

def shuffle(list1, list2):
    temp = list(zip(list1, list2))
    np.random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return res1, res2

def get_images_train_test(pathes, marks, train_size = 0.8, invert=False):
    x = []
    y = []

    for i in range(len(pathes)):
        files = os.listdir(pathes[i])
        for j in range(len(files)):
            image = Image.open(pathes[i] + '\\' + files[j])
            if invert:
                image = ImageOps.invert(image)
            x.append(image)
            y.append(marks[i])

    x, y = shuffle(x, y)
           
    return train_test_split(x, y, train_size=train_size, random_state=42)

def train_speed_dont_decrease_func(epoch_now, decrease_count, train_speed):
    return decrease_count, train_speed

def train_speed_decrease_func(epoch_now, decrease_count, train_speed):
    if (epoch_now % (200 * (decrease_count + 1)) == 0):
             train_speed *= 0.97
             decrease_count += 1
    return decrease_count, train_speed

class functions:
    sigm_coeff = 1
    isru_coeff = 1
    th_coeff = 1
    relulu_coeff = 1
    lu_coeff = 1
    elu_coeff = 1

    def sigm(x):
        return 1 / (1 + np.exp(functions.sigm_coeff * -x))

    def der_sigm(x):
        return functions.sigm(x) * (1 - functions.sigm(x))  


    def relu(x):
        return functions.relu_coeff * x if x >= 0 else 0

    def der_relu(x):
        return functions.relu_coeff if x >= 0 else 0


    def th(x):
        return np.tanh(functions.th_coeff * x)

    def der_th(x):
        return 1 - (functions.th(x) ** 2)
    

class Net:
    def __init__(self, inputs, neurons_array, activations = None, deriviatives = None, weight_border = 0.5, normalize_weights = True):
        if activations != None: 
            self.acts = activations
            self.ders = deriviatives
        else:
            self.acts = [functions.sigm for i in range(len(neurons_array))]
            self.ders = [functions.der_sigm for i in range(len(neurons_array))]
            # self.acts = [functions.sigm for i in range(len(neurons_array))]
            # self.ders = [functions.der_sigm for i in range(len(neurons_array))]

        self.G = [[0 for n in range(neurons_array[i])] for i in range(len(neurons_array))]
        
        self.train_speed_decrease_func = train_speed_dont_decrease_func
        self.round_training_predict = True

        inputs += 1

        self.weights = [[[uniform(-weight_border, weight_border) for k in range(neurons_array[i - 1] + 1 if i > 0 else inputs)] for j in range(neurons_array[i])] for i in range( len(neurons_array))]
        
        if normalize_weights:
            for i in range(len(neurons_array)):
                inputs = neurons_array[i - 1] + 1 if i > 0 else inputs
                b = 0.7 * (neurons_array[i] ** (1 / inputs))
                for j in range(len(self.weights[i])):
                    sum = np.sum(self.weights[i][j])
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] = b * self.weights[i][j][k] / sum

        self.fields = [[0 for j in range(neurons_array[i])] for i in range(len(neurons_array))]
        self.J = []
        self.H = []
        self.train_size_epoch = 1

        self._l = len(self.weights) - 1
    
    def _normalize_weights(self, inputs_count):
        for i in range(len(self.weights)):
                inputs = len(self.weights[i - 1]) + 1 if i > 0 else inputs_count
                b = 0.7 * (len(self.weights[i]) ** (1 / inputs))
                for j in range(len(self.weights[i])):
                    sum = np.sum(self.weights[i][j])
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] = b * self.weights[i][j][k] / sum
        
    def predict(self, signals : any):
        signals = np.append(signals.copy(), 1)

        for i in range(len(self.weights[0])):
            self.fields[0][i] = np.dot(self.weights[0][i], signals)

        for i in range(1, len(self.weights)):
            for j in range(len(self.weights[i])):
                self.fields[i][j] = np.dot(self.weights[i][j], np.append(self.acts[i - 1](np.array(self.fields[i - 1].copy())), 1))

        arr = np.array(self.fields[self._l].copy())
        return self.acts[self._l](arr)
    
    # def predict(self, signals : np.array):
    #     return np.array([self.predict(signals[i]) for i in range(len(signals))])
    
    def fit(self, x_train, y_train, train_speed, max_epochs,  max_error = 0.1, min_hitrate = 0.8):
        errs_history = []
        hitrate_history = []

        decrease_count = 0

        x_train = x_train.copy()
        y_train = y_train.copy()

        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)

        for epoch in range(max_epochs):

            print(f'epoch {epoch + 1} has started', end='')

            x, y = None, None

            x_train, y_train = shuffle(x_train, y_train)

            if self.train_size_epoch == 1:
                x, y = x_train, y_train
            else:
                x, a, y, b = train_test_split(x_train, y_train, train_size=self.train_size_epoch, random_state=42)

            x, y = shuffle(x, y)

            errs = 0
            hitrate = 0

            for i_x in range(len(x)):
                y_pr = self.predict(x[i_x])

                flag = True
                for i in range(len(y_pr)):
                    if self.round_training_predict and np.round(y_pr[i]) != y[i_x][i]:
                        flag = False
                        break
                    elif not self.round_training_predict and y_pr[i] != y[i_x][i]:
                        flag = False
                        break
                if flag:
                    hitrate += 1

                error = y_pr - y[i_x]

                errs += np.square(error).mean()

                self._back_prop(error, x, i_x, train_speed)
            
            errs = errs / len(x) if len(x) > 1 else errs * len(self.weights[self._l])
            hitrate /= len(x)

            errs_history.append(errs)
            hitrate_history.append(hitrate)
            
            print(f', error = {errs}, hitrate = {hitrate}{", aim = " + str(y[i_x]) if len(y) == 1 else ""}')
                
            if min_hitrate != None and hitrate >= min_hitrate:
                print(f'fitted by hitrate')
                return

            if max_error != None and errs <= max_error:
                print(f'fitted by error')
                return

            decrease_count, train_speed = self.train_speed_decrease_func(epoch, decrease_count, train_speed)

        print('ended by epochs')

    def _get_signals(self, x, i, i_x):
        signals = self.acts[i - 1](np.array(self.fields[i - 1].copy())) if i > 0 else x[i_x].copy()
        signals = np.append(signals, 1)
        return signals

    def get_mse_hitrate(self, x_test, y_test):
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)

        mse = 0
        hitrate = 0
        for i in range(len(x_test)):
            y_pr = self.predict(x_test[i])
            #print(y_pr)
            flag = True
            for j in range(len(y_pr)):
                if self.round_training_predict and np.round(y_pr[j]) != y_test[i][j]:
                    flag = False
                    break
                elif not self.round_training_predict and y_pr[j] != y_test[i][j]:
                    flag = False
                    break
            if flag:
                hitrate += 1

            mse += np.square(y_pr - y_test[i]).mean()

        hitrate /= len(x_test)
        mse /= len(x_test)

        return mse, hitrate
    
    def _back_prop(self, error, x, i_x, train_speed):
        # self.__set_gradients(self._l, error)
        # self.__change_weights(self._l, x, i_x, train_speed)

        # for i in range(self._l - 1, -1, -1):
        #     self.__set_gradients(i)
        #     self.__change_weights(i, x, i_x, train_speed)

        for i in range(len(self.weights[self._l])):
            g = self.ders[self._l](self.fields[self._l][i]) * error[i]
            self.G[self._l][i] = g
            signals = self._get_signals(x, self._l, i_x)
            for j in range(len(self.weights[self._l][i])):
                self.weights[self._l][i][j] -= train_speed * self.G[self._l][i] * signals[j]
        
        for i in range(self._l - 1, -1, -1):
            signals = self._get_signals(x, i, i_x)
            for j in range(len(self.weights[i])):
                weighted_sum = 0
                for k in range(len(self.weights[i + 1])):
                    weighted_sum += self.G[i + 1][k] * self.weights[i + 1][k][j]
                g = self.ders[i](self.fields[i][j]) * weighted_sum 

                self.G[i][j] = g

                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= train_speed * self.G[i][j] * signals[k]


    def _set_gradients(self, layer_index, error=None):
        for i in range(len(self.weights[layer_index])):
            g = 0
            if error is None:
                weighted_sum = 0
                for k in range(len(self.weights[layer_index + 1])):
                    weighted_sum += self.G[layer_index + 1][k] * self.weights[layer_index + 1][k][i]
                g = self.ders[layer_index](self.fields[layer_index][i]) * weighted_sum 
            else:
                g = self.ders[layer_index](self.fields[layer_index][i]) * error[i]
            self.G[layer_index][i] = g

    def _change_weights(self, layer_index, x, i_x, train_speed):
        signals = self._get_signals(x, layer_index, i_x)
        for i in range(len(self.weights[layer_index])):
            for j in range(len(self.weights[layer_index][i])):
                self.weights[layer_index][i][j] -= train_speed * self.G[layer_index][i] * signals[j]
        

class NetGC(Net):
    def __init__(self, inputs, neurons_array, activations = None, deriviatives = None, weight_border = 0.5, normalize_weights = True):
        super().__init__(inputs=inputs, neurons_array=neurons_array, activations=activations, deriviatives=deriviatives, weight_border=weight_border, normalize_weights=normalize_weights)
        
    def _set_flat_gradients(self, train_rule):
        self.r = []
        
        for layer in range(self._l):
            signals = self._get_signals([train_rule], layer, 0)
            for i in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][i])):
                    self.r.append(self.G[layer][i] * signals[j])

        self.r = np.array(self.r)

    def _change_weights(self, flat_r, train_speed):
        count = 0
        for i in range(self._l):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= flat_r[count] * train_speed
                    count += 1
    
    def fit(self, x_train, y_train, train_speed, max_epochs,  max_error = 0.1, min_hitrate = 0.8):
        errs_history = []
        hitrate_history = []

        decrease_count = 0

        fitted_flag = False

        x_train = x_train.copy()
        y_train = y_train.copy()

        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)

        for epoch in range(max_epochs):
            print(f'epoch {epoch + 1} has started', end='')

            x, y = None, None

            x_train, y_train = shuffle(x_train, y_train)

            if self.train_size_epoch == 1:
                x, y = x_train, y_train
            else:
                x, a, y, b = train_test_split(x_train, y_train, train_size=self.train_size_epoch, random_state=42)

            x, y = shuffle(x, y)

            errs = 0
            hitrate = 0

            mse_new = 0

            for i_x in range(len(x)):
                mse_old = self.get_mse_hitrate(x, y)[0]

                y_pr = self.predict(x[i_x])

                flag = True
                for i in range(len(y_pr)):
                    if self.round_training_predict and np.round(y_pr[i]) != y[i_x][i]:
                        flag = False
                        break
                    elif not self.round_training_predict and y_pr[i] != y[i_x][i]:
                        flag = False
                        break
                if flag:
                    hitrate += 1

                error = y_pr - y[i_x]

                errs += np.square(error).mean()
                
                self._set_gradients(self._l, error)
                for i in range(self._l - 1, -1, -1):
                    self._set_gradients(i)

                self._set_flat_gradients(x_train[i_x])

                sigma_old = np.dot(self.r, self.r)

                max_descent_iteration = 6

                for i in range(max_descent_iteration):
                    old_w = self.weights
                    self._change_weights(self.r, train_speed)
                    mse_new = self.get_mse_hitrate(x, y)[0]
                    if mse_new >= mse_old:
                        self.weights = old_w
                        break
                    
                old_r = self.r    

                self._set_flat_gradients(x_train[i_x])

                sigma_new = np.dot(self.r, self.r)
                
                beta = sigma_new / sigma_old

                self.r = self.r + beta * old_r

                for i in range(max_descent_iteration):
                    old_w = self.weights
                    self._change_weights(self.r, train_speed)
                    mse_new = self.get_mse_hitrate(x, y)[0]
                    if mse_new >= mse_old:
                        self.weights = old_w
                        break

                if fitted_flag:
                    break
            
            errs = errs / len(x) if len(x) > 1 else errs * len(self.weights[self._l])
            hitrate /= len(x)

            errs_history.append(errs)
            hitrate_history.append(hitrate)
            
            print(f', error = {errs}, hitrate = {hitrate}{", aim = " + str(y[i_x]) if len(y) == 1 else ""}')
                
            if min_hitrate != None and hitrate > min_hitrate:
                print(f'fitted by hitrate')
                return

            if (max_error != None and errs < max_error) or fitted_flag:
                print(f'fitted by error')
                return

            decrease_count, train_speed = self.train_speed_decrease_func(epoch, decrease_count, train_speed)

        print('ended by epochs')

class ElmanNet(Net):
    def __init__(self, inputs, neurons_array, activations=None, deriviatives=None, weight_border=0.5, normalize_weights=True):
        super().__init__(inputs, neurons_array, activations, deriviatives, weight_border, normalize_weights)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i])):
                        self.weights[i][j].append(np.random.uniform(-0.5, 0.5))
        self.recurrent_signals = [[None for j in range(len(self.weights[i]))] for i in range(len(self.weights))]
        self.recurrent_signals_old = [[None for j in range(len(self.weights[i]))] for i in range(len(self.weights))]

        self._normalize_weights(inputs)

    def clear_recurrent_signals(self):
        self.recurrent_signals = [[None for j in range(len(self.weights[i]))] for i in range(len(self.weights))]
        self.recurrent_signals_old = [[None for j in range(len(self.weights[i]))] for i in range(len(self.weights))]

    def predict(self, signals: any):
        signals = np.append(signals.copy(), 1)

        for i in range(len(self.weights[0])):
            without_rec = len(signals)
            self.fields[0][i] = np.dot(self.weights[0][i][:without_rec], signals)
            if self.recurrent_signals_old[0][i] != None:
                self.fields[0][i] += np.dot(self.weights[0][i][without_rec:], self.recurrent_signals[0]) #self.recurrent_signals[0][i] * self.weights[0][i][without_rec]
            self.recurrent_signals[0][i] = self.acts[0](self.fields[0][i].copy())

        for i in range(1, len(self.weights)):
            for j in range(len(self.weights[i])):
                without_rec = len(self.weights[i - 1]) + 1
                #print(f'without_rec = {without_rec}, w_all = {self.weights[i][j]}, w = {self.weights[i][j][without_rec]}')
                self.fields[i][j] = np.dot(self.weights[i][j][:without_rec], np.append(self.acts[i - 1](np.array(self.fields[i - 1])), 1))
                if self.recurrent_signals_old[i][j] != None:
                    self.fields[i][j] += np.dot(self.weights[i][j][without_rec:], self.recurrent_signals[i])
                self.recurrent_signals[i][j] = self.acts[i](self.fields[i][j].copy())

        arr = np.array(self.fields[self._l].copy())
        return self.acts[self._l](arr)

    def fit(self, x_train, y_train, train_speed, max_epochs,  max_error = 0.1, min_hitrate = 0.8):
        errs_history = []
        hitrate_history = []

        decrease_count = 0

        x_train = x_train.copy()
        y_train = y_train.copy()

        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)

        for epoch in range(max_epochs):
            print(f'epoch {epoch + 1} has started', end='')

            x, y = None, None

            x_train, y_train = shuffle(x_train, y_train)

            if self.train_size_epoch == 1:
                x, y = x_train, y_train
            else:
                x, a, y, b = train_test_split(x_train, y_train, train_size=self.train_size_epoch, random_state=42)

            x, y = shuffle(x, y)

            errs = 0
            hitrate = 0

            self.clear_recurrent_signals()

            for i_x in range(len(x)):
                y_pr = self.predict(x[i_x])

                flag = True
                for i in range(len(y_pr)):
                    if self.round_training_predict and np.round(y_pr[i]) != y[i_x][i]:
                        flag = False
                        break
                    elif not self.round_training_predict and y_pr[i] != y[i_x][i]:
                        flag = False
                        break
                if flag:
                    hitrate += 1

                error = y_pr - y[i_x]

                errs += np.square(error).mean()

                self._back_prop(error, x, i_x, train_speed, True if i_x == 0 else False)
                self.recurrent_signals_old = self.recurrent_signals.copy()
            
            errs = errs / len(x) if len(x) > 1 else errs * len(self.weights[self._l])
            hitrate /= len(x)

            errs_history.append(errs)
            hitrate_history.append(hitrate)
            
            print(f', error = {errs}, hitrate = {hitrate}{", aim = " + str(y[i_x]) if len(y) == 1 else ""}')
                
            if min_hitrate != None and hitrate >= min_hitrate:
                print(f'fitted by hitrate')
                return

            if max_error != None and errs <= max_error:
                print(f'fitted by error')
                return

            decrease_count, train_speed = self.train_speed_decrease_func(epoch, decrease_count, train_speed)

        print('ended by epochs')

    def _back_prop(self, error, x, i_x, train_speed, is_firt_rule):
        for i in range(len(self.weights[self._l])):
            g = self.ders[self._l](self.fields[self._l][i]) * error[i]
            self.G[self._l][i] = g
            signals = self._get_signals(x, self._l, i_x)
            l = len(signals)
            for j in range(l):
                self.weights[self._l][i][j] -= train_speed * self.G[self._l][i] * signals[j]
            if not is_firt_rule:
                for j in range(len(self.weights[self._l])):
                    self.weights[self._l][i][l + j] -= train_speed * self.G[self._l][i] * self.recurrent_signals_old[self._l][j]
        
        for i in range(self._l - 1, -1, -1):
            signals = self._get_signals(x, i, i_x)
            for j in range(len(self.weights[i])):
                weighted_sum = 0
                for k in range(len(self.weights[i + 1])):
                    weighted_sum += self.G[i + 1][k] * self.weights[i + 1][k][j]
                g = self.ders[i](self.fields[i][j]) * weighted_sum 

                self.G[i][j] = g

                l = len(signals)
                for k in range(l):
                    self.weights[i][j][k] -= train_speed * self.G[i][j] * signals[k]
                if not is_firt_rule:
                    for k in range(len(self.weights[i])):
                        self.weights[i][j][l + k] -= train_speed * self.G[i][j] * self.recurrent_signals_old[i][k]