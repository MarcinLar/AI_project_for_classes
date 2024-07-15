import numpy as np # Wykorzystywane do przetwarzania i manipulacji tablic danych oraz obliczeń matematycznych w operacjach związanych z uczeniem maszynowym.
import matplotlib.pyplot as plt # Wykorzystywane do tworzenia wykresów i wizualizacji wyników treningu oraz testowania modelu.
import json # Wykorzystywane do wczytywania danych z plików Json do analizy i przetwarzania.
import pandas as pd # Wykorzystywane do wczytywania,przetwarzania i manipulacji danych w formacie DataFrame, pochodzących z plików json
from datetime import datetime # wykorzystywanie do odmierzania czasu treniungu oraz nazewnictwa wykresów
import os # wykorzystywane do zapisu generowanych wykresów do folderu charts


#ładowanie danych z pliku JSON, przetwarzanie do formatu DataFrame. Normalizacja kolumn ask i bid, oraz zamiana kodów walut(string) na unikalne wartości typu int
def load_and_process_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f) #wczytanie danych z pliku

    df = pd.DataFrame(data)#konwersja danych do DataFrame
    df = df.drop(columns=['currency', 'effectiveDate'])# Usunięcie niepotrzebnych kolumn
    df['bid'] = df['bid'] / 10# Skalowanie wartości kolum bid i ask do wartości z zakresu 0 - 1 
    df['ask'] = df['ask'] / 10
    currency_codes = df['code'].unique()#zmiana kodów walut ze stringów na unikalne wartości loczbowe
    code_to_int = {code: idx for idx, code in enumerate(currency_codes)}# mapowanie kodów walut na liczby całkowite
    df['code'] = df['code'].map(code_to_int)# zamiana kodów walut na liczby całkowite w DataFrame

    return df, code_to_int

# implementacja warstwy sieci neuronowej z propagacją do przodu i wstecz, oraz regularyzacją wag i odchyleń
class Layer:
    def __init__(self, number_inputs, number_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(number_inputs, number_neurons)# inicjalizacja wag losowymi wartościami mnożone przez 0.01 aby uniknąć zbyt dużych wartości
        self.biases = np.zeros((1, number_neurons))# inicjalizacja odchyleń 
        #inicjalizacja parametrów regularyzacji
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    #implementacja metody forward 
    def forward(self, inputs):
        self.inputs = inputs# przepisanie danych do wykorzystania w propagacji wstecznej
        self.output = np.dot(inputs, self.weights) + self.biases #obliczenie jako iloczyn skalarny macierzy wejść i wag, oraz dodanie odchyleń
    #implementacja metody backward
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)# obliczanie gradientów wag jako iloczyn skalarny transponowanej macierzy wejść i gradientów z następnej warstwy
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) # obliczanie gradientów odchyleń jako sumy gradientów z następnej warstwy, sumowane wzdłóż osi wejść
        # aktualizacja gradientów wag l1 jeśli współczynnik jest większy od 0. Dodanie do gradientów wag składnika, który jest macierzą wartości 1 lub -1  zależnie od znaku wagi
        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dl1
        #aktualizacja gradientów wag l2 jeśli współczynnik jest więszky od 0. 
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # aktualizacja gradientów odchyleń l1 jeśli współczynnik jest większy od 0. Dodanie do gradientów wag składnika, który jest macierzą wartości 1 lub -1  zależnie od znaku wagi
        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1
        #aktualizacja gradientów odchyleń l2 jeśli współczynnik jest więszky od 0. 
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)#obliczanie gradientów strat jako iloczyn skalarny macierzy gradientów z następnej warstwy i transponowanej macierzy wag

# implementacja funkcj aktywacji ReLU oraz jej pochodnej
class Activation_ReLU:
    #implementacja metody propagacji w przód
    def forward(self, inputs):
        self.inputs = inputs#zapisanie wejść do późniejszego wykorzystania podczas propagacji wstecznej
        self.output = np.maximum(0, inputs)#obliczenie wyjść jako maksymalna wartość pomiędzy 0 a wartością wejściową dla każdej komórki(ujemne wartości są zastępowane zerami)
    # implementacja metody propagacji wstecznej
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()#kopiowanie gradientów z następnej warstwy, aby można było je zmodyfikować
        self.dinputs[self.inputs <= 0] = 0# wyzerowanie gradientów dla nieaktywnych neuronów

# implementacja funkcji aktywacji softmax 
class Activation_Softmax:
    # implementacja metody propagacji w przód 
    def forward(self, inputs):
        self.inputs = inputs#zapisanie wejść do późniejszego wykorzystania podczas propagacji wstecznej
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))# obliczanie wartości wykładniczych, odjęcie od wartości wejściowych maksymalnej wartości z każdej próbki
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)#obliczanie prawdopodobieństw, normalizacja wartości wykładniczych przez sumę wykładniczych dla każdej próbki
        self.output = probabilities#zapisanie wyników prawdopodobieństwa jako wyjścia

    # implementacja metody propagacji wstecznej
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)# inicjalizacja gradientów wyjściowych, utworzenie macierzy gradientów wyjściowych o takiej strukturze jak gradienty z następnej warstwy
        #obliczanie gradientów dla każdej próbki
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):#iterowanie przez próbki danych wyjściowych i odpowiadających im gradienty z następnej warstwy
            single_output = single_output.reshape(-1, 1)#przekształcenie wyjść próbki do formatu kolumnowego
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)#obliczanie macierzy jakobiego jako różnica między diagonalną macierzą wyjść a iloczynem skalarnym macierzy wyjść i jej transpozycji
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)#obliczanie gradientów dla próbek jao iloczyn skalarny macierzy jakobiego i gradientów z następnej warstwy.

#implementacja klasy bazowej dla funkcji strat, pozwala oszacować jak dobrze radzi sobie model.
class Loss:
    #obliczanie średniej straty dla danych wyjściowych modelu i oczekiwanych wartości(y).
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)#obliczanie strat dla każdej próbki 
        data_loss = np.mean(sample_losses)#obliczanie średniej straty jako średniej arytmetycznej strat dla wszystkich próbek
        return data_loss
    
#implementacja klasy z funkcją straty kategorycznej entropii krzyżowej
class Loss_CategoricalCrossentropy(Loss):
    #obliczanie straty krzyżowej entropii dla predykcji i prawdziwych etykiet
    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)#liczba przewidzianych wartości 
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)# obcinanie predykcji aby uniknąć problemu z log(0) podczas obliczania strat
        #obliczanie prawdopodobieństw dla prawidłowych klas 
        if len(y_true.shape) == 1:
            correct_confidences = y_prediction_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_prediction_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)#obliczanie strat jako ujemy logarytm prawdopodobieństw dla prawidłowych klas
        return negative_log_likelihoods
    #obliczanie gradientów dla propagacji wstecznej
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        #obliczanie gradientów jako -y_ture/dvalues a następnie normalizowane przez liczbę próbek
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
    #obliczanie dodatkowej stray regularyzacyjnej dla danej warstwy 
    def regularization_loss(self,layer):
        regularization_loss = 0
        # dodawanie strat regularyzacyjnych L1 i L2 dla wag i odchyleń. W zależności od wartosci współczynników regularyzacji
        if layer.weight_regularizer_l1 >0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l1 >0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    
#implementacja klasy wykorzystującej funkcje Aktywacji Softmax z funkcją straty kategorycznej entropii krzyżowej 
class Activation_Softmax_Loss_CategoricalCrossentropy:
    #inicjalizacja obiektów Activation_Softmax() oraz Loss_CategoricalCrossentropy()
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    # wykonanie propagacji w przód przez funkcję aktywacji softmax i obliczanie straty krzoyżowej entropii
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)#propagacja danych w przód przez funkcję softmax
        self.output = self.activation.output#obliczanie straty na podstawie wyjścia z softmax i prawdzywych etykiet
        return self.loss.calculate(self.output, y_true)
    #wykonanie propagacji wstecznej oraz obliczenie gradientów w odniesieniu do danych wejściowych
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #przygotowanie etykiet, konwersja do indeksów klas
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #obliczanie gradientów jako różnica między predykcjami a prawdziwymi etykietami oraz normalizacja przez liczbę próbek
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

#implementacja mechanizmu optymalizacji modelu sieci neuronowej, wykorzystując algorytm Adam(Adaptive Moment Estimation).
class Oprimizer:
    #inicjalizacja parametrów optymalizatora, początkowa szybkość uczenia, współczynnik zaniku, epsilon oraz beta1 i beta2
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    #obliczanie szybkości uczenia z uwzględnieniem zaniku przed aktualizacją parametrów
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    #aktualizacja parametrów warstwy(wag i odchyleń)
    def update_params(self, layer):
        # inicjalizacja cache i momentum. Jeśli warstwa nie posiada jescze tych parametrów dla wag i odchyleń, Są one inicjalizowane jako zerowe macierze o tych samych kształtach co wagi i odchylenia 
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        #aktualizacja momentum dla pierwszego rzędu(beta1)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        #poprawione momentum pierwszego rzędu
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        #aktualizacja cache drugiego rzędu(beta2)
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        #poprawione cache drugiego rzędu 
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        #aktualizacja wag i odchyleń z użyciem poprawionych momentum i cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    #inkrementacja licznika iteracji po aktualizacji parametrów 
    def post_update_params(self):
        self.iterations += 1

#generowanie wykresów słupkowych dla przebiegów testowych sieci
def plot_loss_and_accuracy(ax, loss, accuracy, title):
    epochs = range(len(loss))
    bar_width = 0.35
    r1 = np.arange(len(epochs))
    r2 = [x + bar_width for x in r1]
    
    ax.bar(r1, loss, color='blue', width=bar_width, label='Test Loss')
    ax.bar(r2, accuracy, color='orange', width=bar_width, label='Test Accuracy')
    
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.legend()

#zapisywanie wygenerowanych wykresuów w folderze charts w formacie png
def save_plots(history_loss, history_accuracy, test_loss, test_accuracy, test_lossbefore, test_accuracybefore, filename_prefix):
    plt.figure(figsize=(12, 24))
    # Wykres z loss
    plt.subplot(3, 1, 1)
    plt.plot(history_loss, label='Loss',color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Wykres z accuracy
    plt.subplot(3, 1, 2)
    plt.plot(history_accuracy, label='Accuracy',color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Wykres test_lossbefore i test_accuracybefore
    plt.subplot(3, 2, 5)
    plot_loss_and_accuracy(plt.gca(), test_lossbefore, test_accuracybefore, 'Test Loss and Accuracy before training')

    plt.subplot(3, 2, 6)
    plot_loss_and_accuracy(plt.gca(), test_loss, test_accuracy, 'Test Loss and Accuracy after training')

    plt.tight_layout()
    plt.subplots_adjust(left=0.03, bottom=0.033, right=0.987, top=0.97, wspace=0.124, hspace=0.27)

    timestamp = datetime.now().strftime("%H_%M_%d_%m_%Y")
    directory = "charts"
    filename = os.path.join(directory, f"{filename_prefix}_{timestamp}.png")
    plt.savefig(filename)
    plt.close()

def main():
    # Wczytywanie i przetwarzanie danych
    df_train, code_to_int_train = load_and_process_data('testing_2024-06-01_2024-06-14_c_data.json') #testing_2024-06-01_2024-06-14_c_data training_2004-01-01_2024-06-16_c_data training_2014-01-01_2024-06-01_c_data Usd_NotUsd_2014-01-01_2024-06-18_c_data
    # Łączenie kolumn w jedną macierz
    data_matrix = df_train[['ask', 'bid', 'code']].values
    # Przemieszanie danych
    np.random.shuffle(data_matrix)
    # Podział danych na 80% treningowe i 20% testowe
    train_size = int(0.8 * len(data_matrix))
    train_data = data_matrix[:train_size]
    test_data = data_matrix[train_size:]
    
    # Rozdzielanie danych na X i y
    X_train = train_data[:, :2]
    y_train = train_data[:, 2].astype(int)
    X_test = test_data[:, :2]
    y_test = test_data[:, 2].astype(int)
    # Definiowanie modelu
    layer1  = Layer(X_train.shape[1], 64, weight_regularizer_l1=3e-4, bias_regularizer_l1=3e-4 ,weight_regularizer_l2=4e-4, bias_regularizer_l2=4e-4)# utworzenie pierwszej warstwy  z X_train,shape[1] czyli 2 wejściami oraz 64 neuronami (wyjściami), z regularizacją L1 i L2.
    activation1 = Activation_ReLU()# utworzenie funkcji aktywacji ReLU dla pierwszej warstwy.
    layer2 = Layer(64, 128)# utworzenie drugiej warstwy z 128 neuronami
    activation2 = Activation_ReLU()# utworzenie funkcji aktywacji ReLU dla drugiej warstwy.
    layer3 = Layer(128, 128)# utworzenie trzeciej warstwy z 128 neuronami.
    activation3 = Activation_ReLU()# utworzenie funkcji aktywacji ReLU dla trzeciej warstwy.
    layer4 = Layer(128, 64)#utworzenie czwartej warstwy z 64 neuronami
    activation4 = Activation_ReLU()# utworzenie funkcji aktywacji ReLU dla czwartej warstwy.
    layer5 = Layer(64, 13)# utworzenie ostatniej warstwy, wyjściowej z 13 neuronami (liczba klas wyjściowych). Dla danych Usd_NotUsd należy ustawić liczbę klasy wyjściowych na 2
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()#utworzenie funkcji aktywacji Softmax i straty kategorycznej entropii krzyżowej.
    optimizer = Oprimizer(learning_rate=0.02,decay=5e-7)# utworzenie optymalizatora  z określoną szybkością uczenia i współczynnikiem zaniku. 

    # utworzenie zmiennych do przechowywania danych potrzebnych do generowania wykresów
    history_loss = []
    history_accuracy = []
    test_loss = []
    test_accuracy = []
    test_lossbefore = []
    test_accuracybefore = []

    #Testowanie modelu przed treningiem
    layer1.forward(X_test)#przeprowadzenie progpagacji w przód przez pierwszą warstwę
    activation1.forward(layer1.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami pierwszej warstwy
    layer2.forward(activation1.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji pierwszej warstwy
    activation2.forward(layer2.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami drugiej warstwy
    layer3.forward(activation2.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji drugiej warstwy
    activation3.forward(layer3.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami trzeciej warstwy
    layer4.forward(activation3.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji trzeciej warstwy
    activation4.forward(layer4.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami czwartej warstwy
    layer5.forward(activation4.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji czwartej warstwy
    loss = loss_activation.forward(layer5.output, y_test)#obliczanie straty za pomocą funkcji aktywacji Softmax i straty kategorycznej entropii krzyżowej
    predictions = np.argmax(loss_activation.output, axis=1)#przewidywanie klasy poprzez wybór indeksu z najwyższym prawdopodobieństwem
    accuracy = np.mean(predictions == y_test)# Obliczanie dokładność jako średnia prawidłowych przewidywań
    test_lossbefore.append(loss)#zapisanie danych do wygenerowania wykresów
    test_accuracybefore.append(accuracy)#zapisanie danych do wygenerowania wykresów
    print(f'validation before training, acc: {accuracy:.3f}, loss: {loss:.3f}')# wyświetlenie wyników w konsoli

    #Trening modelu  
    start_time = datetime.now()# czas rozpoczęcia treningu
    for epoch in range(5001): # określenie ile epok ma proces trenowania
        layer1.forward(X_train)#przeprowadzenie progpagacji w przód przez pierwszą warstwę
        activation1.forward(layer1.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami pierwszej warstwy
        layer2.forward(activation1.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji pierwszej warstwy
        activation2.forward(layer2.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami drugiej warstwy
        layer3.forward(activation2.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji drugiej warstwy
        activation3.forward(layer3.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami trzeciej warstwy
        layer4.forward(activation3.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji trzeciej warstwy
        activation4.forward(layer4.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami czwartej warstwy
        layer5.forward(activation4.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji czwartej warstwy
        data_loss = loss_activation.forward(layer5.output,y_train)#obliczanie straty za pomocą funkcji aktywacji Softmax i straty kategorycznej entropii krzyżowej
        regularization_loss = loss_activation.loss.regularization_loss(layer1) + loss_activation.loss.regularization_loss(layer2) + loss_activation.loss.regularization_loss(layer3) + loss_activation.loss.regularization_loss(layer4) +loss_activation.loss.regularization_loss(layer5)# obliczanie straty regularyzacyjnej dla wszystkich warstw 
        loss = data_loss + regularization_loss# obliczanie całkowitej straty jako sumę straty danych i straty regularyzacyjnej

        predictions = np.argmax(loss_activation.output, axis=1)#przewidywanie klasy poprzez wybór indeksu z najwyższym prawdopodobieństwem
        accuracy = np.mean(predictions == y_train)# Obliczanie dokładność jako średnia prawidłowych przewidywań
        history_loss.append(loss)#zapisanie danych do wygenerowania wykresów
        history_accuracy.append(accuracy)#zapisanie danych do wygenerowania wykresów
        
        # wypisanie co 100 epok danych  dokładności, straty całkowitej, straty danych, straty regularyzacji i aktualnej szybkość uczenia.
        if not epoch % 100:
            print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}, data_loss:{data_loss:.3f}, reg_loss:{regularization_loss:.3f}, lr: {optimizer.current_learning_rate}')
        
        loss_activation.backward(loss_activation.output, y_train)#przeprowadzenie propagacji wstecznej przez warstwę straty
        layer5.backward(loss_activation.dinputs)# przeprowadzenie propagacji wstecznej przez ostatnią warstwę(wyjściową)
        activation4.backward(layer5.dinputs)# przeprowadzenie propagacji wstecznej przez funkcję aktywacji ReLU czwartej warstwy
        layer4.backward(activation4.dinputs)# przeprowadzenie propagacji wstecznej przez czwartą warstwę
        activation3.backward(layer4.dinputs)# przeprowadzenie propagacji wstecznej przez funkcję aktywacji ReLU trzeciej warstwy
        layer3.backward(activation3.dinputs)# przeprowadzenie propagacji wstecznej przez trzecią warstwę
        activation2.backward(layer3.dinputs)# przeprowadzenie propagacji wstecznej przez funkcję aktywacji ReLU drugiej warstwy
        layer2.backward(activation2.dinputs)# przeprowadzenie propagacji wstecznej przez drugą warstwę
        activation1.backward(layer2.dinputs)# przeprowadzenie propagacji wstecznej przez funkcję aktywacji ReLU pierwszej warstwy
        layer1.backward(activation1.dinputs)# przeprowadzenie propagacji wstecznej przez pierwszą warstwę
        optimizer.pre_update_params()# przygotowanie optymalizatora do aktualizacji parametrów
        optimizer.update_params(layer1)# Aktualizacja parametrów dla pierwszej warstwy.
        optimizer.update_params(layer2)# Aktualizacja parametrów dla drugiej warstwy.
        optimizer.update_params(layer3)# Aktualizacja parametrów dla trzeciej warstwy.
        optimizer.update_params(layer4)# Aktualizacja parametrów dla czwartej warstwy.
        optimizer.update_params(layer5)# Aktualizacja parametrów dla ostatniej warstwy.
        optimizer.post_update_params()# zakończenie aktualizaji parametrów.
    
    end_time = datetime.now()# czas zakończenia treningu

    # Testowanie po treningu
    layer1.forward(X_test)#przeprowadzenie progpagacji w przód przez pierwszą warstwę
    activation1.forward(layer1.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami pierwszej warstwy
    layer2.forward(activation1.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji pierwszej warstwy
    activation2.forward(layer2.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami drugiej warstwy
    layer3.forward(activation2.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji drugiej warstwy
    activation3.forward(layer3.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami trzeciej warstwy
    layer4.forward(activation3.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji trzeciej warstwy
    activation4.forward(layer4.output)#przeprowadzenie progpagacji w przód przez funkcję aktywacji ReLU z wynikami czwartej warstwy
    layer5.forward(activation4.output)#przeprowadzenie progpagacji w przód przez drugą warstwę z wynikami funkcji aktywacji czwartej warstwy
    loss = loss_activation.forward(layer5.output, y_test)#obliczanie straty za pomocą funkcji aktywacji Softmax i straty kategorycznej entropii krzyżowej
    predictions = np.argmax(loss_activation.output, axis=1)#przewidywanie klasy poprzez wybór indeksu z najwyższym prawdopodobieństwem
    accuracy = np.mean(predictions == y_test)# Obliczanie dokładność jako średnia prawidłowych przewidywań
    test_loss.append(loss)#zapisanie danych do wygenerowania wykresów
    test_accuracy.append(accuracy)#zapisanie danych do wygenerowania wykresów
    print(f'validation after training, acc: {accuracy:.3f}, loss: {loss:.3f}')# wyświetlenie wyników w konsoli

    #wyświetlenie godziny rozpoczęczia, godziny zakończenia treningu oraz ich różnicy
    print(f"Training started at: {start_time.strftime('%H:%M:%S')}")
    print(f"Training ended at: {end_time.strftime('%H:%M:%S')}")
    print(f"Training duration: {end_time - start_time}")

    #opracowywanie danych w formacie wykresów
    plt.figure(figsize=(12, 24))
    # Wykres ze stratą
    plt.subplot(3, 1, 1)
    plt.plot(history_loss, label='Loss',color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Wykres z dokładonością 
    plt.subplot(3, 1, 2)
    plt.plot(history_accuracy, label='Accuracy',color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Wykresy z danymi testowymi przed treningiem i po treningu 
    plt.subplot(3, 2, 5)
    plot_loss_and_accuracy(plt.gca(), test_lossbefore, test_accuracybefore, 'Test Loss and Accuracy before training')

    plt.subplot(3, 2, 6)
    plot_loss_and_accuracy(plt.gca(), test_loss, test_accuracy, 'Test Loss and Accuracy after training')

    plt.tight_layout()
    plt.subplots_adjust(left=0.03, bottom=0.033, right=0.987, top=0.97, wspace=0.124, hspace=0.27)
    plt.show()
    save_plots(history_loss, history_accuracy, test_loss, test_accuracy, test_lossbefore, test_accuracybefore, 'wykresy')

if __name__ == "__main__":
    main()