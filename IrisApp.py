from DataLoader import DataLoader
from ModelManager import ModelManager
from IrisGUI import IrisGUI

class IrisApp:
    def __init__(self):
        self.loader = DataLoader()
        self.manager = None

    def start(self):
        # 1. تحميل واستكشاف البيانات
        self.loader.load_data().explore().plot_data()
        
        # 2. تقييم وتدريب النماذج
        self.manager = ModelManager(self.loader.X, self.loader.y)
        self.manager.evaluate_multiple_models().train_final_model()
        
        # 3. تشغيل الواجهة الرسومية
        print("\n loading...")
        gui = IrisGUI(self.manager)
        gui.build_gui()
        gui.run()

if __name__ == "__main__":
    app = IrisApp()
    app.start()