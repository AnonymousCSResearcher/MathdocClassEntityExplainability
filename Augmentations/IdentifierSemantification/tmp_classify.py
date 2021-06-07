from paths import root_path,spec_path,spec_file,sub_class
# root and specific path, specific filename, subject classes
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

full_path = root_path + "/" + spec_path

# LOAD FORMULA FEATURES

formula_nr_limit = 3495 # smallest class 'astro-ph'

# get feature vectors and labels
x = []
y = []
for spec_class in sub_class:
    with open(full_path + "/" + spec_class + "/" + spec_file,"rb") as f:
        data = pickle.load(f)
        x.extend(data[0:formula_nr_limit])
        y.extend([spec_class for i in range(0,formula_nr_limit)])

# CLASSIFY FORMULAE

# split train and test set
x_train,x_test,y_train,y_test = train_test_split(x,y)

# init classifier model
model = LogisticRegression()

# fit and evaluate model
#model = model.fit(x_train,y_train)
#score = model.score(x_test,y_test)
score = cross_val_score(model,x,y,cv=10)
print(score)

# PLOT FORMULA SPACE

# dimensionality reduction
pca = PCA()
x_red = pca.fit_transform(x)
# label binarize
y_bin = label_binarize(y,classes=sub_class)

# scatter plot with colorbar
fig,ax = plt.subplots()
im = ax.scatter(x_red[:,0],x_red[:,1],c=y_bin)
fig.colorbar(im,ax=ax)

plt.show()

print("end")