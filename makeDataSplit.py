from monai.data import load_decathlon_datalist

PATH = "C:/awilde/britta/LTU/DataMining/Data/Task01_BrainTumour/dataset.json"

datalist = load_decathlon_datalist(PATH, True, "training")

print(len(datalist))

train_size = int(len(datalist) * 0.8)
val_size = int(len(datalist) * 0.10)
test_size = len(datalist) - train_size - val_size

print(train_size, val_size, test_size)

train_data = datalist[:train_size]
val_data = datalist[train_size:train_size+val_size]
test_data = datalist[train_size+val_size:]


print(len(train_data))
print(len(val_data))
print(len(test_data))
