# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:28:19 2021

@author: ashum
"""


from skimage import io
import pandas as pd



#import os 

import os 

#get a list of files from file path

pathforall= r'C:\Users\ashum\OneDrive\Desktop\Leukemia\archive\C-NMC_Leukemia\training_data\fold_0\all'
dir_list=os.listdir(pathforall)

pathforhem= r'C:\Users\ashum\OneDrive\Desktop\Leukemia\archive\C-NMC_Leukemia\training_data\fold_0\hem'
dir_list2=os.listdir(pathforhem)


df = pd.DataFrame()


for i in dir_list[:300]:
    img=io.imread(pathforall + "\\" + i)
 
    my_df = pd.DataFrame(img.flatten()).transpose()
    #my_df.columns = ['image_data']
    #my_df = pd.DataFrame(my_df)
    #my_df = my_df.query('image_data>0')
    #my_df.columns = my_df.iloc[0] 
    #my_df = my_df[1:]
    #my_df["mean"]=my_df.loc[:,0].mean()
    #my_df["std"]=my_df.iloc[:,-1].std()
    my_df['Class'] = 'ALL'
    #my_df=my_df[["mean","std"]]
    my_df=my_df.head(n=1)
    df = df.append(my_df, ignore_index=True)
    

#df.hist(bins=10)
#plt.show()

df2 = pd.DataFrame()

for i in dir_list2[:300]:
    img2=io.imread(pathforhem + "\\" + i)
 
    my_df2 = pd.DataFrame(img2.flatten()).transpose()
    #my_df2["mean"]=my_df2.loc[:,0].mean()
    #my_df2["std"]=my_df2.iloc[:,-1].std()
    my_df2['Class'] = 'Not ALL'
    #my_df2=my_df2[["mean","std",]]
    my_df2=my_df2.head(n=1)
    df2 = df2.append(my_df2, ignore_index=True)

#df2.hist(bins=10)
#plt.show()

df_stack = pd.concat([df, df2])

#df_stack['gmean'] = (df_stack["mean"]*df_stack["var"])**(1/2)

#Store class as y and rest of data as x
X = df_stack.drop('Class', axis=1)
y = df_stack['Class']

#split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

#make predictions

y_pred = svclassifier.predict(X_test)

#metrics

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))















    


 
