from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelManager:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = SVC(gamma='auto')  
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)

    def evaluate_multiple_models(self):
        # تقسيم البيانات
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=1)
        
        models = [('LR', LogisticRegression(max_iter=200)), ('KNN', KNeighborsClassifier()), 
                  ('CART', DecisionTreeClassifier()), ('SVC', SVC(gamma='auto'))]
        
        print("\n--- the end result (Cross-Validation) ---")
        for name, model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        return self

    def train_final_model(self):
        self.model.fit(self.X_train, self.y_train)
        prediction = self.model.predict(self.X_test)
        print(f'\nTest Accuracy: {accuracy_score(self.y_test, prediction)}')
        print(f'Classification Report: \n {classification_report(self.y_test, prediction)}')
        return self

    def predict(self, input_data):
        return self.model.predict([input_data])[0]