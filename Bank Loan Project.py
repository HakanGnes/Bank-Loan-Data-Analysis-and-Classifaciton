#####################################################
# Bank Term Deposit Marketing Data Analysis and Classifaciton
#####################################################

# Data Description:
# The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.
#
# Domain:Banking
#
# Context:
# This case is about a bank (Thera Bank) whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.
#
# Learning Outcomes:
#
# Exploratory Data Analysis
# Preparing the data to train a model
# Training and making predictions using a classification model
# Model evaluation
# Objective:
# The classification goal is to predict the likelihood of a liability customer buying personal loans.
#
# Steps and tasks:
#
# Read the column description and ensure you understand each attribute well
# Study the data distribution in each attribute, share your findings
# Get the target column distribution.
# Split the data into training and test set in the ratio of 70:30 respectively
# Use different classification models (Logistic, K-NN and Naïve Bayes) to predict the likelihood of a liability customer buying personal loans
# Print the confusion matrix for all the above models
# Give your reasoning on which is the best model in this case and why it performs better?
# References:
#
# Data analytics use cases in Banking
# Machine Learning for Financial Marketing

# Data Link : https://www.kaggle.com/datasets/krantiswalke/bank-personal-loan-modelling


# GENEL BAKIS
import lazypredict
from lazypredict.Supervised import LazyClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ROCAUC, ClassificationReport
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
colors= ['#00876c','#85b96f','#f7e382','#f19452','#d43d51']

def load():
    df = pd.read_csv("Data/Bank_Personal_Loan_Modelling.csv")
    return df

df = load()
df.head()

df = df.drop(labels = ["ID","ZIP Code"],axis=1)
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Değişken İsimlendirilmesi
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

# Outlier Analizi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# Korelasyon Analizi
corr = df[num_cols].corr()
corr

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap=colors, annot=True, linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()

# Tüm Değişkenler vs Target Değişken Analizi
featuresAndTarget = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage','Personal Loan']
features = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']

target = 'Personal Loan'

fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,15), dpi=100)

for i in range(len(features)):
    x = i//2
    y = i % 2
    sns.countplot(x=features[i] , data=df , ax=ax[x,y])
    ax[x,y].set_xlabel(features[i], size = 8)
    ax[x,y].set_title('{} vs. {}'.format(target, features[i]), size = 15)

plt.show()

# Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, True)

def cat_summary(dataframe, col_name):
    summary_df = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.xticks(rotation=90)
    return summary_df

outputs = []

plt.figure(figsize=(15, 12))

for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i+1)
    summary_df = cat_summary(df, col)
    outputs.append(summary_df)

plt.tight_layout()

fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=outputs[0].values, colLabels=outputs[0].columns, cellLoc='center', loc='center')
plt.show()


# Sayısal Değişken Analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50, figsize=(9,5))
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, True)


def num_summary2(dataframe, numerical_col, ax=None, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot and ax:
        dataframe[numerical_col].hist(bins=50, ax=ax)
        ax.set_xlabel(numerical_col)
        ax.set_title(numerical_col)

fig, axs = plt.subplots(len(num_cols), 1, figsize=(9, 5 * len(num_cols)))

for i, col in enumerate(num_cols):
    num_summary2(df, col, ax=axs[i], plot=True)

plt.tight_layout()
plt.show()
#Distrubiton Analysis
def num_plot(dataframe, col):
    color = "#85b96f"
    plt.figure(figsize=(10, 5))
    sns.histplot(x=dataframe[col], color=color, label=col)

    # Plotting the mean age line
    mean = df[col].mean()
    plt.axvline(x=mean, color='black', linestyle="--", label=dataframe[col].mean())

    plt.legend()
    plt.title('Distribution')
    plt.show()


for col in num_cols:
    num_plot(df,col)


def num_plot(dataframe, col, ax):
    color = "#85b96f"
    sns.histplot(x=dataframe[col], color=color, label=col, ax=ax)

    # Plotting the mean age line
    mean = dataframe[col].mean()
    ax.axvline(x=mean, color='black', linestyle="--", label=f"Mean: {mean:.2f}")

    ax.legend()
    ax.set_title(f'Distribution - {col}')

fig, axs = plt.subplots(len(num_cols), 1, figsize=(10, 5 * len(num_cols)))

for i, col in enumerate(num_cols):
    num_plot(df, col, ax=axs[i])

plt.tight_layout()
plt.show()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Personal Loan", cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

standart_scaler = StandardScaler()
df["Age"] = standart_scaler.fit_transform(df[["Age"]])
df["Experience"] = standart_scaler.fit_transform(df[["Experience"]])
df["Income"] = standart_scaler.fit_transform(df[["Income"]])
df.head()


# Modelleme

X = df.drop('Personal Loan_1', axis=1)
y = df['Personal Loan_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


lcf = LazyClassifier(predictions = True)
models, predictions = lcf.fit(X_train, X_test, y_train, y_test)
models

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train,verbose=False)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, pred))

# Yellowbrick Raporu
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
plt.suptitle("Classification Reports", family='Serif', size=15, ha='center', weight='bold')

# ROC Curve
axs[0].set_title('ROC Curve')
roc_visualizer = ROCAUC(model, classes=[0, 1], ax=axs[0])
roc_visualizer.fit(X_train, y_train)
roc_visualizer.score(X_test, y_test)

# Sınıflandırma Raporu
axs[1].set_title('Classification Report')
classification_visualizer = ClassificationReport(model, classes=[0, 1], support=True, ax=axs[1], cmap=colors)
classification_visualizer.fit(X_train, y_train)
classification_visualizer.score(X_test, y_test)

plt.figtext(0.05, -0.05, "Observation: Logistic Regression performed well with an accuracy score of 81%",
            family='Serif', size=14, ha='left', weight='bold')

plt.tight_layout()
plt.show()
