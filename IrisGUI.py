import tkinter as tk
from tkinter import messagebox

class IrisGUI:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.root = tk.Tk()
        self.root.title("Iris Classification Project")
        self.entries = []

    def build_gui(self):
        labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        for i, text in enumerate(labels):
            tk.Label(self.root, text=text).grid(row=i, column=0, padx=10, pady=5)
            e = tk.Entry(self.root)
            e.grid(row=i, column=1, padx=10, pady=5)
            self.entries.append(e)
        
        tk.Button(self.root, text=" flower classification ", command=self.perform_prediction, bg="blue", fg="white").grid(row=4, columnspan=2, pady=10)

    def perform_prediction(self):
        try:
            inputs = [float(e.get()) for e in self.entries]
            result = self.model_manager.predict(inputs)
            messagebox.showinfo("the tesult", f"the flower is {result}")
        except:
            messagebox.showerror("Errorr", " Please enter the right numbers" )

    def run(self):
        self.root.mainloop()