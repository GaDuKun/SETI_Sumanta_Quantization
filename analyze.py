import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

dir_path = 'trained_models/'
progress_data = []
# Load the training history from a pkl file
for filename in os.listdir(dir_path):
    if filename.endswith('pkl'):
        with open(os.path.join(dir_path,filename), 'rb') as f:
            data = pickle.load(f)
            progress_data.append((filename,data))

# remove prefix and postfix:
prefix = 'trainedResnet_'
postfix = '.pkl'

#plot training accuracy
plt.figure()
for filename,data in progress_data:
    train_acc = [x * 100 for x in data['accuracy']]
    train_loss = data['loss']
    epochs = range(len(train_loss))
    label = filename[len(prefix):-len(postfix)]
    plt.plot(epochs, train_acc, label = label) 
plt.title('Training accuracy for multiple models')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

#plot training Loss
plt.figure()
for filename,data in progress_data:
    train_acc = [x * 100 for x in data['accuracy']]
    train_loss = data['loss']
    epochs = range(len(train_loss))
    label = filename[len(prefix):-len(postfix)]
    plt.plot(epochs, train_loss, label = label)   
plt.title('Training Loss for multiple models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()