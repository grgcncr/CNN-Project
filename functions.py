from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models


def confusion_matrix(y: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def get_dataset(root_dir):
    from COVID19Dataset import COVID19Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return COVID19Dataset(root_dir, transform=transform)

def get_resnet50(prossesed):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4),
        nn.Softmax(dim=1)
    )
    if prossesed == True:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from COVID19Dataset import CLASSES

def browse_rootdir():
    directory = filedialog.askdirectory(title="Select Root Directory")
    rootdir_entry.delete(0, tk.END)
    rootdir_entry.insert(0, directory)

def validate_rootdir(rootdir):
    # Check if the rootdir contains all the required class folders
    missing_classes = []
    for cls in CLASSES:
        cls_path = os.path.join(rootdir, cls)
        if not os.path.exists(cls_path):
            missing_classes.append(cls)
    return missing_classes

def submit():
    # Handle the submission of rootdir and selected model
    rootdir = rootdir_entry.get()
    selected_model = model_var.get()
    if not rootdir:
        messagebox.showerror("Error", "Please select a root directory!")
        return
    if selected_model not in ["cnn1", "cnn2", "resnet50", "resnet50_processed"]:
        messagebox.showerror("Error", "Please select a valid model!")
        return
    missing_classes = validate_rootdir(rootdir)
    if missing_classes:
        messagebox.showerror(
            "Error",
            f"The selected directory is missing the following required class folders: {', '.join(missing_classes)}"
        )
        return
    app.selected_data = {"rootdir": rootdir, "model": selected_model}
    app.destroy()

def gui_select_model():
    # Run the GUI and return the selected options
    global app, rootdir_entry, model_var
    app = tk.Tk()
    app.title("Select Model and Path")
    app.geometry("470x200")
    rootdir_label = tk.Label(app, text="Root Directory:")
    rootdir_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    rootdir_entry = tk.Entry(app, width=40)
    rootdir_entry.grid(row=0, column=1, padx=10, pady=10)
    browse_button = tk.Button(app, text="Browse", command=browse_rootdir)
    browse_button.grid(row=0, column=2, padx=10, pady=10)
    model_label = tk.Label(app, text="Select Model:")
    model_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
    model_var = tk.StringVar(value="cnn1")  # Default selection
    model_menu = tk.OptionMenu(app, model_var, "cnn1", "cnn2", "resnet50", "resnet50_processed")
    model_menu.grid(row=1, column=1, padx=10, pady=10)
    submit_button = tk.Button(app, text="Submit", command=submit)
    submit_button.grid(row=2, column=1, pady=20)
    app.selected_data = None  # Placeholder for selected data
    app.mainloop()
    return app.selected_data

