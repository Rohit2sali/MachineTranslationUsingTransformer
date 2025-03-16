import pickle
import pickle
class data:
    def __init__(self):
        with open(r"/all_data/final_train_data.pkl", "rb") as f:
            self.train_data = pickle.load(f)
        with open(r"/all_data/test_data.pkl", "rb") as f:
            self.test_data = pickle.load(f)
        with open(r"/all_data/val_data.pkl", "rb") as f:
            self.val_data = pickle.load(f)
        with open(r"/all_data/final_target_train_data.pkl", "rb") as f:
            self.target_train_data = pickle.load(f)
        with open(r"/all_data/target_test_data.pkl", "rb") as f:
            self.target_test_data = pickle.load(f)
        with open(r"/all_data/target_val_data.pkl", "rb") as f:
            self.target_val_data = pickle.load(f)

    def get_data(self):
        return self.train_data, self.val_data, self.test_data, self.target_train_data, self.target_val_data, self.target_test_data
