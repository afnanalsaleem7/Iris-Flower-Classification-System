import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class DataLoader:
    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None

    def load_data(self):
        # تحميل البيانات لضمان عمل الكود دون الحاجة لروابط خارجية
        iris = load_iris()
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        self.dataset = pd.DataFrame(data=iris.data, columns=self.feature_names)
        self.dataset['class'] = [self.target_names[i] for i in iris.target]
        
        self.X = self.dataset.drop(['class'], axis=1)
        self.y = self.dataset['class']
        print("the data been loaded")
        return self

    def explore(self):
        # عرض المعلومات الأساسية
        print(f"Shape: {self.dataset.shape}")
        print(self.dataset.describe())
        print(self.dataset['class'].value_counts())
        return self

    def plot_data(self):
        # الرسوم البياني  
        sns.set_palette('husl')
        
        # رسم Violin plots لكل الخصائص
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, col in enumerate(self.feature_names):
            sns.violinplot(y='class', x=col, data=self.dataset, inner='quartile', ax=axes[i//2, i%2])
        plt.tight_layout()
        plt.show()

        # رسم Pairplot
        sns.pairplot(self.dataset, hue='class', markers='+')
        plt.show()
        return self