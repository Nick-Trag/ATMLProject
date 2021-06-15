
@author: victoria
"""

#εισαγωγη βιβλιοθηκών 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics,model_selection
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss 

#φόρτωση αρχείου 
data_final = pd.read_excel(r'C:\Users\victo\OneDrive\Υπολογιστής\data_final.xlsx')

#έλεγχος για ποσοστό null τιμών 
null_final = data_final.isnull().sum().sort_values(ascending=False) 
not_null_final = data_final.notnull().sum().sort_values(ascending=False) 
percentage_not_null_final = 100.000-data_final.isna().mean().round(4) * 100
data_missing_values_final = pd.concat({'Null': null_final, 'Not Null': not_null_final,
                                    'Percentage': percentage_not_null_final}, axis=1)

#αφαίρεση στηλών με null τιμές πάνω απο το 90%
data_final=data_final.drop(['Patient ID','D-Dimer','Urine - Sugar','Urine - Sugar','Mycoplasma pneumoniae',
                      'Partial thromboplastin time (PTT) ',
                      'Prothrombin time (PT), Activity','Fio2 (venous blood gas analysis)',
                      'Urine - Nitrite','Vitamin B12','Lipase dosage','Albumin',
                      'Phosphor','Arteiral Fio2','Ferritin','ctO2 (arterial blood gas analysis)',
                      'Arterial Lactic Acid','Hb saturation (arterial blood gases)',
                      'pCO2 (arterial blood gas analysis)','Base excess (arterial blood gas analysis)',
                      'pH (arterial blood gas analysis)','Total CO2 (arterial blood gas analysis)',
                      'HCO3 (arterial blood gas analysis)','pO2 (arterial blood gas analysis)',
                      'Magnesium','Ionized calcium ','Urine - Ketone Bodies','Urine - Protein',
                      'Urine - Esterase','Urine - Hyaline cylinders','Urine - Urobilinogen',
                      'Urine - Granular cylinders','Urine - Leukocytes','Urine - Bile pigments',
                      'Urine - Crystals','Urine - Aspect','Urine - pH','Urine - Hemoglobin',
                      'Urine - Color','Urine - Yeasts','Urine - Density','Urine - Red blood cells',
                      'Relationship (Patient/2)','Myeloblasts','Myelocytes','Metamyelocytes',
                      'Promyelocytes','Segmented','Rods #','Lactic Dehydrogenase',
                      'Creatine phosphokinase (CPK) ','International 2ized ratio (INR)',
                      'Base excess (venous blood gas analysis)',
                      'Hb saturation (venous blood gas analysis)',
                      'pCO2 (venous blood gas analysis)','pO2 (venous blood gas analysis)',
                      'Total CO2 (venous blood gas analysis)','pH (venous blood gas analysis)',
                      'HCO3 (venous blood gas analysis)','Alkaline phosphatase','Gamma-glutamyltransferase ',
                      'Indirect Bilirubin','Direct Bilirubin','Total Bilirubin',
                      'Serum Glucose','help'],axis=1)

#1 για γέμισμα κενών με στοιχείων με 1 και 2 για γέμισμα με την μέση τιμή της κάθε στήλης
m = input("do you want to continue with fillna0(press=1) or fillnamean(press=2): ")
m= float(m)
if m==1:
    data_final = data_final.fillna(0)
if m==2:
    data_final.fillna(data_final.mean(), inplace=True)
    
#ορισμός δεδομέων εκπαίδευσης και ελέγχου
X = data_final.drop('SARS-Cov-2 exam result',axis=1)
y=data_final['SARS-Cov-2 exam result']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.4,random_state=1)   

#εκπτύπωση αριτθμού τιμών στις 2 κλάσεις
print(y_train.value_counts())

#κανονικοποίηση
minMaxScaler = preprocessing.MinMaxScaler()
x_train = minMaxScaler.fit_transform(X_train)
x_test = minMaxScaler.transform(X_test)

#επιλογή τεχνικής απο τον χρήστη 
m = input("if you want to continue with class imbalance press 0,smote press 1,smote+tomek link press 2, nearmiss press 3 : ")
m= float(m)

#------------------------------------------------------------------------------
#με class imbalance 
if m==0:

    sns.set(style="darkgrid")
    fig_2 = plt.figure(figsize=(12, 6), dpi=600)
    plt.title('Classes of Values')
    ax = sns.countplot(x=y_train.values.ravel())
    plt.show()
    
    #επιλογή αλγορίθμου:
    m = input("do you want to continue with KNN (1) or Decision tree (2): ")
    m= float(m)
    #knn
    if m==1: 
        
        #gridSearch για εύρεση καλύτερων παραμέτρων 
        '''
        parameters = {'weights':['uniform', 'distance'],}
        grid = GridSearchCV(KNeighborsClassifier(), parameters, refit = True, verbose = 3, scoring = "accuracy") 
        grid.fit(x_train, y_train)
        print("Best estimators parameters")
        print(grid.best_estimator_)
        '''
        classifierKNN = KNeighborsClassifier(n_neighbors=10,weights='distance',metric='minkowski',p=2)
        classifierKNN.fit(x_train, y_train)
        y_pred = classifierKNN.predict(x_test)
        print("KNN_recall_before_smote: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
        print("KNN_precision_before_smote: %2f" % metrics.precision_score(y_test,y_pred,average='macro',zero_division=0))
        print("KNN_accurany_before_smote: %2f" % metrics.accuracy_score(y_test,y_pred))
        print("KNN_f1_before_smote: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
        c_matrix = confusion_matrix(y_test,y_pred)
        print(c_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(c_matrix, cmap='Blues')
        fig.colorbar(plot)
        plt.show()
        
        #Decision tree
    if m==2: 
        #gridSearch για εύρεση καλύτερων παραμέτρων 
        '''
        parameters = {'criterion':['gini', 'entropy']}
        grid = GridSearchCV(DecisionTreeClassifier(), parameters, refit = True, verbose = 3, scoring = "accuracy") 
        grid.fit(X_train, y_train)
        print("Best estimators parameters")
        print(grid.best_estimator_)
        '''
        classifierDT = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=7,random_state=1)
        classifierDT.fit(x_train,y_train)
        y_pred = classifierDT.predict(x_test)
        print("DTrecall_before_smote: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
        print("DTprecision_before_smote: %2f" % metrics.precision_score(y_test,y_pred,average='macro',zero_division=0))
        print("DTaccurany_before_smote: %2f" % metrics.accuracy_score(y_test,y_pred))
        print("DTf1_before_smote: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
        c_matrix = confusion_matrix(y_test,y_pred)
        print(c_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(c_matrix, cmap='Blues')
        fig.colorbar(plot)
        plt.show()
#------------------------------------------------------------------------------    
#με smote
if m==1:
    oversample = SMOTE(sampling_strategy='minority', random_state=1, k_neighbors=10)
    x_over, y_over = oversample.fit_resample(x_train, y_train.values.ravel())

    sns.set(style="darkgrid")
    fig_2 = plt.figure(figsize=(12, 6), dpi=600)
    plt.title('Classes of Values after SMOTE')
    ax = sns.countplot(x=y_over)
    
    m = input("do you want to continue with KNN (1) or Decision tree (2): ")
    m= float(m)
    
    #knn
    if m==1:
        classifierKNN = KNeighborsClassifier(n_neighbors=10,weights='distance',metric='minkowski',p=2)
        classifierKNN.fit(x_over, y_over)
        y_pred = classifierKNN.predict(x_test)
        print("KNNrecall_after_smote_tomek_link: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
        print("KNNprecision_after_smote_tomek_link: %2f" % metrics.precision_score(y_test,y_pred,average='macro'))
        print("KNNaccurany_after_smote_tomek_link: %2f" % metrics.accuracy_score(y_test,y_pred))
        print("KNNf1_after_smote_tomek_link: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
        c_matrix = confusion_matrix(y_test,y_pred)
        print(c_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(c_matrix, cmap='Blues')
        fig.colorbar(plot)
        plt.show()
    
    #decision tree
    if m==2: 
        classifierDT = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=7,random_state=1)
        classifierDT.fit(x_over,y_over)
        y_pred = classifierDT.predict(x_test)
        print("DTrecall_after_smote: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
        print("DTprecision_after_smote: %2f" % metrics.precision_score(y_test,y_pred,average='macro',zero_division=0))
        print("DTaccurany_after_smote: %2f" % metrics.accuracy_score(y_test,y_pred))
        print("DTf1_after_smote: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
        c_matrix = confusion_matrix(y_test,y_pred)
        print(c_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(c_matrix, cmap='Blues')
        fig.colorbar(plot)
        plt.show()
#------------------------------------------------------------------------------
#με smote + tomek links
if m==2:
    oversample = SMOTE(sampling_strategy='minority', random_state=1, k_neighbors=10)
    x_over, y_over = oversample.fit_resample(x_train, y_train.values.ravel())
    undersample = TomekLinks(sampling_strategy='all')
    x_under, y_under = undersample.fit_resample(x_over, y_over)
 
    sns.set(style="darkgrid")
    fig_2 = plt.figure(figsize=(12, 6), dpi=600)
    plt.title('Classes of Values after SMOTE and Tomek Links')
    ax = sns.countplot(x=y_under)
    
    m = input("do you want to continue with KNN (1) or Decision tree (2): ")
    m= float(m)
    
    if m==1:
        classifierKNN = KNeighborsClassifier(n_neighbors=10,weights='distance',metric='minkowski',p=2)
        classifierKNN.fit(x_under, y_under)
        y_pred = classifierKNN.predict(x_test)
        print("KNNrecall_after_smote_tomek_link: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
        print("KNNprecision_after_smote_tomek_link: %2f" % metrics.precision_score(y_test,y_pred,average='macro'))
        print("KNNaccurany_after_smote_tomek_link: %2f" % metrics.accuracy_score(y_test,y_pred))
        print("KNNf1_after_smote_tomek_link: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
        c_matrix = confusion_matrix(y_test,y_pred)
        print(c_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(c_matrix, cmap='Blues')
        fig.colorbar(plot)
        plt.show()

    if m==2: 
        
        classifierDT = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=7,random_state=1)
        classifierDT.fit(x_under,y_under)
        y_pred = classifierDT.predict(x_test)
        print("DTrecall_after_smote_tomek_link: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
        print("DTprecision_after_smote_tomek_link: %2f" % metrics.precision_score(y_test,y_pred,average='macro',zero_division=0))
        print("DTaccurany_after_smote_tomek_link: %2f" % metrics.accuracy_score(y_test,y_pred))
        print("DTf1_after_smote_tomek_link: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
        c_matrix = confusion_matrix(y_test,y_pred)
        print(c_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(c_matrix, cmap='Blues')
        fig.colorbar(plot)
        plt.show()     
#------------------------------------------------------------------------------    
#με nearmiss-3
if m==3:
   nr = NearMiss(version=3) 
   x_near, y_near= nr.fit_sample(x_train, y_train.ravel()) 

   sns.set(style="darkgrid")
   fig = plt.figure(figsize=(12, 6), dpi=600)
   plt.title('Classes of Values after nearmiss')
   ax = sns.countplot(x=y_near)
   
   m = input("do you want to continue with KNN (1) or Decision tree (2): ")
   m= float(m)
   
   if m==1:
       classifierKNN = KNeighborsClassifier(n_neighbors=10,weights='distance',metric='minkowski',p=2)
       classifierKNN.fit(x_near, y_near)
       y_pred = classifierKNN.predict(x_test)
       print("KNN_recall_after_nearmiss: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
       print("KNN_precision_after_nearmiss: %2f" % metrics.precision_score(y_test,y_pred,average='macro',zero_division=0))
       print("KNN_accurany_after_nearmiss: %2f" % metrics.accuracy_score(y_test,y_pred))
       print("KNN_f1_after_nearmiss: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
       c_matrix = confusion_matrix(y_test,y_pred)
       print(c_matrix)
       fig = plt.figure()
       ax = fig.add_subplot(111)
       plot = ax.matshow(c_matrix, cmap='Blues')
       fig.colorbar(plot)
       plt.show()
       
   if m==2: 
       classifierDT = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=7,random_state=1)
       classifierDT.fit(x_near,y_near)
       y_pred = classifierDT.predict(x_test)
       print("DTrecall_after_nearmiss: %2f" % metrics.recall_score(y_test,y_pred,average='macro'))
       print("DTprecision_after_nearmiss: %2f" % metrics.precision_score(y_test,y_pred,average='macro',zero_division=0))
       print("DTaccurany_after_nearmiss: %2f" % metrics.accuracy_score(y_test,y_pred))
       print("DTf1_after_nearmiss: %2f" % metrics.f1_score(y_test,y_pred,average='macro'))
       c_matrix = confusion_matrix(y_test,y_pred)
       print(c_matrix)
       fig = plt.figure()
       ax = fig.add_subplot(111)
       plot = ax.matshow(c_matrix, cmap='Blues')
       fig.colorbar(plot)
       plt.show()
         