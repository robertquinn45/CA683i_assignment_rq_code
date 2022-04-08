#######################
# Author: Robert Quinn
# CA683i Data Analytics Data Mining Assignment
#######################
# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
!pip install kds
import kds
from google.colab import drive
drive.mount('/content/drive')
import os #, sys
#sys.path.insert(0,nb_path)
os.chdir("/content/drive/My Drive/CA683i_data_analytics_mining_assignment")
def plot_lift_curve(y_val, y_pred, step=0.01):  # function referenced from website: https://howtolearnmachinelearning.com/code-snippets/lift-curve-code-snippet/
    
    #Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    #Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_val
    aux_lift['predicted'] = y_pred
    #Order the values for the predicted probability column:
    aux_lift.sort_values('predicted',ascending=False,inplace=True)
    
    #Create the values that will go into the X axis of our plot
    x_val = np.arange(step,1+step,step)
    #Calculate the ratio of ones in our data
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    #Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []
    
    #Calculate for each x value its correspondent y value
    for x in x_val:
        num_data = int(np.ceil(x*len(aux_lift))) #The ceil function returns the closest integer bigger than our number 
        data_here = aux_lift.iloc[:num_data,:]   # ie. np.ceil(1.4) = 2
        ratio_ones_here = data_here['real'].sum()/len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)
           
   #Plot the figure
    fig, axis = pyplot.subplots()
    fig.figsize = (40,40)
    axis.plot(x_val, y_v, 'g-', linewidth = 3, markersize = 5)
    axis.plot(x_val, np.ones(len(x_val)), 'k-')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    pyplot.title('Lift Curve')
    pyplot.show()

train_full_df = pd.read_csv('bank-additional-full.csv', sep=';')
test_full_df=pd.read_csv('bank-additional.csv', sep=';')
print('The shape of train_full_df: %d x %d' % train_full_df.shape)
#print(train_full_df.head)
print('The shape of test_full_df: %d x %d' % test_full_df.shape)
#print(test_full_df.head)
# merge both dataframe togheter for analysis of full customer base
df_full_cust_base=pd.concat([train_full_df, test_full_df])
print('The shape of df_full_cust_base: %d x %d' % df_full_cust_base.shape)
print(df_full_cust_base.columns)
### This gives us the probability of each occurance
print('This gives us the probability of each occurance')
print(df_full_cust_base['y'].value_counts(1))
df_full_cust_base_drp=df_full_cust_base.drop(columns = 'duration')  # dropping "duration" of the call column as this indicates call length and we willnot durationof call in advance
    # and call duration (ie. 0 seconds) is a good indictaion of the outcome(target)
#split dataset in features and target variable
print('The shape of df_full_cust_base_drp: %d x %d' % df_full_cust_base_drp.shape)
# print(df_full_cust_base_drp.columns)
# find out how many rows thare are in the full dataset with at least one null value column on the row
print('find out how many rows thare are in the full dataset with at least one null value column on the row')
print(df_full_cust_base_drp.shape[0] - df_full_cust_base_drp.dropna().shape[0])
print('The shape of df_full_cust_base_drp: %d x %d' % df_full_cust_base_drp.shape)
# RECOMMEDNED TO DROP NULL VALUED ROWS AS IT CONFUSES THE DECISION TREE - there are no null values in the full customer base dataset
# No missing values.
df_full_cust_base_drp.info()
# Encoding part 1: after converting the y column from yes/no to 1/0
df_full_cust_base_drp['y'] = df_full_cust_base_drp.y.map(dict(yes=1, no=0))
# Encoding part 2: create dummy variables for all the categorical variables, replacing the original columns with all encoded (grouped named) columns
df_full_cust_base_encod = pd.get_dummies(df_full_cust_base_drp)
#encoding complete

y = df_full_cust_base_encod.y # Target variable
#print(y.head())
X = df_full_cust_base_encod.drop(columns = 'y') # X will have all other independent variables/features apart from the target/outcome y
#print(X.columns)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y) # 70% training and 30% test, random_state:  when we fix the seed for the pseudorandom number generator, we get an identical split of the original dataset
#stratify=y guarantees that equal percentage of targets (y) and non-targets are in both training and test datasets
#In order to validate properly your model, the class distribution should be constant along with the different splits (train, validation, test)
print('printing y_train set target/non-target totals')  # ensure equal percentage of targets and no n-targets in each train and test datsets
print(y_train.value_counts(0))
print('printing y_test set target/non-target totals')  # ensure equal percentage of targets and no n-targets in each train and test datsets
print(y_test.value_counts(0))
print('printing X_train set count of rows')  # ensure equal percentage of targets and no n-targets in each train and test datsets
print(X_train.shape[0])
print('printing X_test set count of rows')  # ensure equal percentage of targets and no n-targets in each train and test datsets
print(X_test.shape[0])

# Recursive Feature Elimination
# feature extraction
model = DecisionTreeClassifier(max_depth=6)
rfe = RFE(model, n_features_to_select=1) # prints the full ordering of the features
fit = rfe.fit(X_train, y_train)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
# summarize all features
#for i in range(X_train.shape[1]):
#	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
    
#===========================================================================
# now print out the features in order of ranking
#===========================================================================
print('now print out the features in order of ranking')
from operator import itemgetter
features = X_train.columns.to_list()
for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):
    print('|',x,y)
    #print('Rank: %d, feature_name %s' % (x, y))
    
#===========================================================================
# ok, this time let's choose the top 35 features and use them for the model
#===========================================================================
print('this time lets choose the top 35 features and use them for the model')
n_features_to_select = 35
rfe = RFE(model, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

#===========================================================================
# use the model to predict the target savings leads for the test data
#===========================================================================
print('use the model to predict the target savings leads for the test data')
predictions = rfe.predict(X_test)

#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({"Id":X_test.index, y:predictions})
output.to_csv('RFE_test_datset_feature_output.csv', index=False)

# Model Accuracy after RFE, how often is the classifier correct?
print("Accuracy after running RFE for decision tree with maximum 6 tree length:",metrics.accuracy_score(y_test, predictions)) # the percentage of this output figure is the classification rate (Accuracy: 0.8599279040682705 returned)
print('Precision Score after running RFE for decision tree with maximum 6 tree length: ', precision_score(y_test, predictions))
# The classifier only detects X% of potential clients that will suscribe to a term deposit.
print('Recall Score after running RFE for decision tree with maximum 6 tree length: ', recall_score(y_test, predictions))
print('F1 Score after running RFE for decision tree with maximum 6 tree length: ', f1_score(y_test, predictions))
cmrf = confusion_matrix(y_test, predictions)
print('confusion_matrix after running RFE for decision tree with maximum 6 tree length: ',cmrf)

# The model is X% sure that the potential client will suscribe to a term deposit. 
# The model is only retaining X% of clients that agree to suscribe a term deposit.
print('Precision Score for RFE: ', precision_score(y_test, predictions))
# The classifier only detects X% of potential clients that will suscribe to a term deposit.
print('Recall Score for RFE: ', recall_score(y_test, predictions))
print('F1 Score for RFE: ', f1_score(y_test, predictions))

# Building Decision Tree Model
# Let's create a Decision Tree Model using Scikit-learn.
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Evaluating Model
#Let's estimate, how accurately the classifier or model can predict the savings account lead
#Accuracy can be computed by comparing actual test set values and predicted values.

# Model Accuracy, how often is the classifier correct?
print("Accuracy of decision tree with maximum tree length:",metrics.accuracy_score(y_test, y_pred)) # the percentage of this output figure is the classification rate
# accuracy can be imporved by tuning the parameters in the Decision Tree Algorithm
print('Precision Score for decision tree with maximum tree length: ', precision_score(y_test, y_pred))
# The classifier only detects X% of potential clients that will suscribe to a term deposit.
print('Recall Score of decision tree with maximum tree length: ', recall_score(y_test, y_pred))
from sklearn.metrics import f1_score
print('F1 Score of decision tree with maximum tree length: ', f1_score(y_test, y_pred))
cmrf = confusion_matrix(y_test, y_pred)
print('confusion_matrix for decision tree with maximum tree length: ',cmrf)

print(list(zip(X.columns, clf.feature_importances_)))


# get feature importance
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# print(X.columns)
# create a list of the independent variables/feature dataframe column names
feature_cols=X.columns.values.tolist()

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=6) # 6 max depth with default criterion "gini" has better accuracy 0.9025 than 3,4,5,7 max tree depth # criterion="entropy" was tested, but slightly lower accuracy 0.9021 than gini 0.9025

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of decision tree with maximum 6 tree length:",metrics.accuracy_score(y_test, y_pred)) # the percentage of this output figure is the classification rate
# accuracy can be imporved by tuning the parameters in the Decision Tree Algorithm
print('Precision Score for decision tree with maximum 6 tree length: ', precision_score(y_test, y_pred))
# The classifier only detects X% of potential clients that will suscribe to a term deposit.
print('Recall Score of decision tree with maximum 6 tree length: ', recall_score(y_test, y_pred))
print('F1 Score of decision tree with maximum 6 tree length: ', f1_score(y_test, y_pred))
cmrf = confusion_matrix(y_test, y_pred)
print('confusion_matrix for decision tree with maximum 6 tree length: ',cmrf)

print ("dec tree prediction data type is: ",y_pred.dtype)
np.savetxt('decision_tree_test_predictions.csv', y_pred, delimiter =' ')
print(list(sorted(zip(clf.feature_importances_,X.columns)))) # reverse=True sorts in descending order

#Visualizing Decision Trees
from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('savings_account_leads_max_trees.png')
Image(graph.create_png())

print('Random Forest Classifier execution')
# RandomForestClassifier Model Exceution and accuracy and confusion matrix generation
rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
y_predrf = rf.predict(X_test)
cmrf = confusion_matrix(y_test, y_predrf)
accrf = metrics.accuracy_score(y_test, y_predrf)
print('RandomForestClassifier confusion_matrix is: ',cmrf)
print('RandomForestClassifier accuracy score is: ',accrf)
print('RandomForestClassifier Precision Score: ', precision_score(y_test, y_predrf))
print('RandomForestClassifier Recall Score: ', recall_score(y_test, y_predrf))
print('RandomForestClassifier F1 Score: ', f1_score(y_test, y_predrf))


print('Creating Lift Curve for RandomForestClassifier')
plot_lift_curve(y_test, y_predrf, step=0.01)
print('Lift Curve creation completed')

print('Creating Gain chart for RandomForestClassifier')
kds.metrics.plot_cumulative_gain(y_test, y_predrf)

print ('generating random forest feature importance')
print(list(sorted(zip(rf.feature_importances_,X.columns)))) # reverse=True sorts in descending order

# This pruned model (when setting criterion="entropy" and max_depth=3) is less complex, explainable, and easy to understand than the previous decision tree model plot.

# drop columns that the feature importance from the Gini/max 6 tree depth model execution above deem unimportant and then reexecute the model and generate accuracy scores:

X_train = X_train.drop(columns = [
'previous',
'poutcome_nonexistent',
'month_sep',
'month_nov',
'month_may',
'month_mar',
'month_jul',
'month_dec',
'month_aug',
'marital_unknown',
'marital_single',
'marital_divorced',
'loan_yes',
'loan_unknown',
'loan_no',
'job_unknown',
'job_unemployed',
'job_technician',
'job_student',
'job_self-employed',
'job_entrepreneur',
'job_admin.',
'housing_unknown',
'housing_no',
'emp.var.rate',
'education_unknown',
'education_professional.course',
'education_illiterate',
'education_basic.9y',
'education_basic.6y',
'default_yes',
'default_unknown',
'default_no',
'day_of_week_wed',
'day_of_week_tue',
'day_of_week_thu',
'contact_cellular'
])

X_test = X_test.drop(columns = [
'previous',
'poutcome_nonexistent',
'month_sep',
'month_nov',
'month_may',
'month_mar',
'month_jul',
'month_dec',
'month_aug',
'marital_unknown',
'marital_single',
'marital_divorced',
'loan_yes',
'loan_unknown',
'loan_no',
'job_unknown',
'job_unemployed',
'job_technician',
'job_student',
'job_self-employed',
'job_entrepreneur',
'job_admin.',
'housing_unknown',
'housing_no',
'emp.var.rate',
'education_unknown',
'education_professional.course',
'education_illiterate',
'education_basic.9y',
'education_basic.6y',
'default_yes',
'default_unknown',
'default_no',
'day_of_week_wed',
'day_of_week_tue',
'day_of_week_thu',
'contact_cellular'
])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=6) # 6 max depth with default criterion "gini" has better accuracy 0.9025 than 3,4,5,7 max tree depth # criterion="entropy" was tested, but slightly lower accuracy 0.9021 than gini 0.9025

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy after manually removing the unimportant features:",metrics.accuracy_score(y_test, y_pred)) # accuracy does not change on max depth 6 and Gini index after removing the unimportant features, accruacy remains at 0.9025969248878098
#read https://www.datacamp.com/community/tutorials/decision-tree-classification-python

#scores
cmdt = confusion_matrix(y_test, y_pred)
accdt = metrics.accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier max depth 6 and unimportnant features removed confusion_matrix is: ',cmdt)
print('DecisionTreeClassifier max depth 6 and unimportnant features removed accuracy score is: ',accdt)
print('DecisionTreeClassifier max depth 6 and unimportnant features removed Precision Score: ', precision_score(y_test, y_pred))
print('DecisionTreeClassifier max depth 6 and unimportnant features removed Recall Score: ', recall_score(y_test, y_pred))
print('DecisionTreeClassifier max depth 6 and unimportnant features removed F1 Score: ', f1_score(y_test, y_pred))
