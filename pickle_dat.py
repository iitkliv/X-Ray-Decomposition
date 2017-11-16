from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
import pickle
from glob import glob
import numpy as np
import pandas as pd

def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files)) ):
        print file_name
        print 'Are we in the loop ?'
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 10== 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    #if len(loc_train_labels) > 0:
    #    df = pd.read_csv(loc_train_labels)
    #    return np.array(dataset), np.array(df["Class"])
    #else:
    return np.array(dataset)




Dataa= dir_to_dataset("/home/siplab/50_images_data/*.png","")
# Data and labels are read 

train_set_x = Dataa[:]
#val_set_x = Dataa[31:40]
#test_set_x = Dataa[41:50]
#train_set_y = y[:30]
#val_set_y = y[31:40]
#test_set_y = y[41:50]
# Divided dataset into 3 parts. I had 6281 images.

#train_set = train_set_x, train_set_y
print 'Type of train_set_x',type(train_set_x)
#print train_set_x
#val_set = val_set_x, val_set_y
#test_set = test_set_x, val_set_y

#dataset = [train_set, val_set, test_set]

f = gzip.open('traffic_file.pkl.gz','wb')
pickle.dump(train_set_x, f, protocol=2)
f.close()
