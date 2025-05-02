import matplotlib.pyplot as plt
import os

save_dir = os.path.expanduser("~/HNTS-MRG/results")
os.makedirs(save_dir, exist_ok=True)

# Lire le fichier texte
with open(os.path.join(save_dir,"losses2.txt"), "r") as f:
    log_lines = f.readlines()

# Extraire les données
epochs = []
train_losses = []
val_losses = []

for line in log_lines:
    if not line.strip():
        continue
    parts = line.strip().split(',')
    epoch = int(parts[0].split()[1])
    train_loss = float(parts[1].split(':')[1])
    val_loss = float(parts[2].split(':')[1])

    epochs.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Créer le graphique
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Sauvegarder l'image
plt.savefig(os.path.join(save_dir,"loss_plot2.png"))
print("✅ Le graphique a été sauvegardé dans 'loss_plot2.png'")
