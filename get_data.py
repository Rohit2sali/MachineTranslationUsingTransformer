import pickle
class data:
    def __init__(self):
        path = "C:\machine learning\learningpython\english-german-both.pkl"
        with open(path, "rb") as file:
            self.data = pickle.load(file)

    def get_data(self, no_examples):
        input = []
        output = []
        for i in range(no_examples):
            input.append(self.data[i][0])
            output.append(self.data[i][1])
        train_data = input[:1000]
        test_data = input[1000:1300]
        val_data = input[1300:]
        target_train_data = output[:1000]
        target_test_data = output[1000:1300]
        target_val_data = output[1300:]
        return input, output, train_data, val_data, test_data, target_train_data, target_val_data, target_test_data
