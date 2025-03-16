import pickle
import pickle
class data:
    def __init__(self):
        with open(r"/kaggle/input/datasetfortranslation/final_train_data.pkl", "rb") as f:
            self.train_data = pickle.load(f)
            # self.train_data = self.train_data[:1000]
        with open(r"/kaggle/input/datasetfortranslation/test_data.pkl", "rb") as f:
            self.test_data = pickle.load(f)
            self.test_data = self.test_data[:300]
        with open(r"/kaggle/input/datasetfortranslation/val_data.pkl", "rb") as f:
            self.val_data = pickle.load(f)
            self.val_data = self.val_data[:300]
        with open(r"/kaggle/input/datasetfortranslation/final_target_train_data.pkl", "rb") as f:
            self.target_train_data = pickle.load(f)
        with open(r"/kaggle/input/datasetfortranslation/target_test_data.pkl", "rb") as f:
            self.target_test_data = pickle.load(f)
            self.target_test_data = self.target_test_data[:300]
        with open(r"/kaggle/input/datasetfortranslation/target_val_data.pkl", "rb") as f:
            self.target_val_data = pickle.load(f)
            self.target_val_data = self.target_val_data[:300]

    def get_data(self):
        return self.train_data, self.val_data, self.test_data, self.target_train_data, self.target_val_data, self.target_test_data
