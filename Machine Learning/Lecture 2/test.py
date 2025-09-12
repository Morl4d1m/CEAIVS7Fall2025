import os

folder = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS7\Machine Learning\Lecture 2\dataset1_G_noisy_ASCII"

print("Folder exists:", os.path.isdir(folder))
if os.path.isdir(folder):
    print("Files in folder:")
    for f in os.listdir(folder):
        print("  ", f)