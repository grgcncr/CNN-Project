import torch
from torch.utils.data import random_split, DataLoader
from functions import get_dataset, get_resnet50, gui_select_model
from train_test import train_one_epoch, test
import random
import matplotlib.pyplot as plt
ROOTDIR = None
model = None

if __name__ == "__main__":
    selected_options = gui_select_model()
    if selected_options:
        ROOTDIR = selected_options["rootdir"]
        model = selected_options["model"]
        print(f"\nSelected Root Directory: {ROOTDIR}\n")
        print(f"Selected Model: {model}\n")
        dataset = get_dataset(ROOTDIR)
        random_indexes = random.sample(range(len(dataset)), 25)
        dataset.display_batch(random_indexes)
        # Image count
        class_counts = {cls_name: 0 for cls_name in dataset.classes}
        for _, label in dataset.image_paths:
            class_counts[dataset.classes[label]] += 1
        
        # Chart
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Images per class')
        plt.xlabel('Class')
        plt.ylabel('Image count')
        plt.show()
        print("Images per class:", class_counts,"\n")
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
        batch_size = 64
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        print(f"Train size: {len(train_ds)}")
        print(f"Validation size: {len(val_ds)}")
        print(f"Test size: {len(test_ds)}\n")

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from CNN1 import CNN1
        from CNN2 import CNN2

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learning_rate = 1e-3
        beta1 = 0.9
        beta2 = 0.99
        batch_size = 64
        max_epochs = 20
        patience = 5 
        tolerance = 0.5 

        if model == 'cnn1':
            model = CNN1(num_classes=4).to(device)
        elif model == 'cnn2':
            model = CNN2(num_classes=4).to(device)
        elif model == 'resnet50':
            model = get_resnet50(prossesed=False).to(device)
            learning_rate = 1e-4
            max_epochs = 5
        elif model == 'resnet50_processed':
            model = get_resnet50(prossesed=True).to(device)
            learning_rate = 1e-4
            max_epochs = 5

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        early_stop_counter = 0

        print(f"\n\n--------------- Started training ---------------")
        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            val_loss, val_accuracy, cm = test(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience and abs(train_loss - val_loss) > tolerance:
                print("Early stopping triggered!")
                break
        print(cm)
        model.load_state_dict(torch.load('best_model.pth'))
        print("--------------- Training completed ---------------")




