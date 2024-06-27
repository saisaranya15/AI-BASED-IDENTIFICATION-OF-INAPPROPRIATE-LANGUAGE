import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# A word cloud is a visualization technique to represent text data, where the size of each word corresponds to its frequency in the given text
from wordcloud import WordCloud
from textwrap import wrap
import nltk
import re
from nltk.corpus import stopwords
import numpy as np

df = pd.read_csv('labeled_data.csv')
print(df.shape ) # first five row of data
print(df.head())
# first five row of data

# lost five row of data

# inforamtion about  data
print(df.info())
print(df.columns)
# Data type of  spesific column
print(df.dtypes)
# chech the null value in Data
print(df.isnull().sum())
# Describe the data in Stastical
print(df.describe().T)
# check the duplicate value
print(df.duplicated().T)
# chech uniqu values
print(df.nunique())
# Target column value count
print(df['class'].value_counts())
# This code uses Seaborn to create a countplot of the
# 'label' column from the DataFrame 'df', visualizing the distribution of categories in the textual data.
sns.set(style='darkgrid')  # Set the style for the plot
plt.figure(figsize=(8, 6))  # Set the size of the plot

# Use the `countplot` function from Seaborn to create the plot
sns.countplot(x=df['class'])

plt.title('Countplot of Textual Data')
plt.xlabel('Categories')
plt.ylabel('Count')

plt.show()
# hist plot of Target column
sns.histplot(df['class'])
sns.histplot(df.isnull())
print('___________________________________________________')
print("The null values of dataset" + ' ' + str(df.isnull().sum()))
print('___________________________________________________')


missing_values_count = df.isnull().sum()




# Create a bar plot to visualize missing values
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.pointplot(x=missing_values_count.index, y=missing_values_count.values)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.title('Missing Values by Column')
plt.show()
sns.histplot(df.duplicated())
print('___________________________________________________')
print("The duplicated values of dataset" + ' ' + str(df.duplicated().sum()))
print('___________________________________________________')
# Corilation of data
col = df.select_dtypes(int)
sns.heatmap(col.corr())
# generates a Kernel Density Estimation (KDE) plot for the numerical data in the 'class' column of the Data
sns.kdeplot(df['class'])


print(df)
df["message_length"] = df["tweet"].apply(len)

# Check summary statistics of message lengths
print(df["message_length"].describe())

# Plot the distribution of message lengths
plt.figure(figsize=(8, 6))
sns.histplot(df["message_length"], bins=50)
plt.title("Distribution of Message Lengths")
plt.xlabel("Message Length")
plt.ylabel("Count")
plt.show()
df['class'].value_counts().plot.pie(explode=[0.05, 0.05, 0.05], autopct='%1.1f%%', startangle=90, shadow=True,
                                    figsize=(6, 8))
plt.title('Pie Chart for result')
plt.show()


def wordcloud(data, title):
    wc = WordCloud(width=600, height=530, max_words=150, colormap="Dark2").generate_from_text(data)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('\n'.join(wrap(title, 60)), fontsize=13)
    plt.show()


# Assuming 'text' is the column containing the text data in your DataFrame 'df'
text_data = df['tweet'].str.cat(sep='\n')

# Use str.split() to split the text into lines and remove unnecessary characters
lines = text_data.split("\n")
lines = [line.strip() for line in lines if line.strip() != ""]

# Combine the lines into a single text string
preprocessed_text = " ".join(lines)

wordcloud(preprocessed_text, "Word Cloud for Enron Methanol")
defective_data = df[df['class'] == True]

# Combine the text from 'tweet' column for defective data
defective_text = " ".join(defective_data['tweet'])

# Split the text into words
defective_words = defective_text.split()

# Create a word frequency distribution for defective data
defective_word_freq = pd.Series(defective_words).value_counts()

# Plot the top 20 most common words in defective messages
plt.figure(figsize=(10, 6))
sns.barplot(x=defective_word_freq[:20].index, y=defective_word_freq[:20].values)
plt.title("Top 20 Most Common Words in Defective Messages")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

print(df['tweet'].str.len().hist())
print(df.info())
print(df.head())

print(df['class'].value_counts())
df = df[['tweet', 'class']]
nltk.download('stopwords')
stop = set(stopwords.words('english'))
corpus = []
new = df['tweet'].str.split()
new = new.values.tolist()
corpus = [word for i in new for word in i]


from collections import defaultdict

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word] += 1


def plot_top_stopwords_barchart(text):
    stop = set(stopwords.words('english'))

    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]
    from collections import defaultdict
    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] += 1

    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
    x, y = zip(*top)
    plt.bar(x, y)
    plt.show()


plot_top_stopwords_barchart(df['tweet'])
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
clean_tweet = []
for i in range(0, len(df['tweet'])):
    revi = re.sub('[^a-zA-Z]', ' ', df['tweet'][i])
    revi = revi.lower()
    revi = revi.split()
    revi = [ps.stem(word) for word in revi if not word in stopwords.words('english')]
    revi = ' '.join(revi)
    clean_tweet.append(revi)
print(clean_tweet[1])
df['clean_tweet'] = clean_tweet
print(df.head())
x = df['clean_tweet']
y = df['class']
print(y.value_counts())



# Convert text to binary
from sklearn.feature_extraction.text import HashingVectorizer

hvectorizer = HashingVectorizer(n_features=10000, norm=None, alternate_sign=False, stop_words='english')
x = hvectorizer.fit_transform(x).toarray()



from imblearn.over_sampling import RandomOverSampler

sm = RandomOverSampler()
a, b = sm.fit_resample(x, y)
# plot the pie plot for the y before and after applying the smote technique
plt.figure(figsize=(10, 5))
y.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 5))
b.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.3, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)





from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier




de = DecisionTreeClassifier()
t1 = time.time()
de.fit(x_train[:1000], y_train[:1000])
de_pred = de.predict(x_test)
de_acc = accuracy_score(de_pred, y_test)
cl_repo = classification_report(de_pred, y_test)
t2 = time.time()
print("DecisionTreeClassifier model")
print(de_acc)
print('____________________________________')
print(t2 - t1)
print('_____________________________________')
print(cl_repo)



from sklearn.ensemble import RandomForestClassifier

ra = RandomForestClassifier()
t1 = time.time()
ra.fit(x_train[:1000], y_train[:1000])
ra_pred = ra.predict(x_test)
t2 = time.time()
print("RandomForestClassifier model")
ra_acc = accuracy_score(ra_pred, y_test)
print(ra_acc)
print('________________________________')
print(t2 - t1)
print('_________________________________')
ra_repo = classification_report(ra_pred, y_test)
print(ra_repo)


print(df.head())

inp = ['rt mayasolov woman complain clean hous amp man alway take trash']
inp_b = hvectorizer.fit_transform(inp).toarray()
result = ra.predict(inp_b)
print(result)


if result == 0:
    print('Hate Speech')
elif result == 1:
    print('Not offensive language')
else:
    print('offensive language')


print(df['clean_tweet'][1])
inp1 = ['rt mleew boy dat cold tyga dwn bad cuffin dat hoe st place']
inp_b1 = hvectorizer.fit_transform(inp1).toarray()
result = ra.predict(inp_b1)
print(result)

if result == 0:
    print('Hate Speech')
elif result == 1:
    print('Not offensive language')
else:
    print('offensive language')





from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


model = Sequential()
# add the first LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# add the dropout layer
model.add(Dropout(0.2))

# add the dropout layer
model.add(Dropout(0.2))
# add the third LSTM layer
model.add(LSTM(units=50, return_sequences=True))
# add the dropout layer
model.add(Dropout(0.2))
# add the fourth LSTM layer
model.add(LSTM(units=50))
# add the dropout layer
model.add(Dropout(0.2))
# add the output layer
model.add(Dense(units=3,activation="softmax"))
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# summarize the model
model.summary()
# fit the model
model.fit(x_train[:100], y_train[:100], epochs=10, batch_size=32)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

inp1 = ['rt mleew boy dat cold tyga dwn bad cuffin dat hoe st place']
inp_b1 = hvectorizer.fit_transform(inp1).toarray()
result = model.predict(inp_b1)
print(result)

result=np.argmax(result)

if result == 0:
    print('Hate Speech')
elif result == 1:
    print('Not offensive language')
else:
    print('offensive language')





from json import *

model_json=model.to_json()
with open("model_architecture.json","w") as json_file:
    json_file.write(model_json)


model.save_weights("model_weights.weights.h5")



import joblib


joblib.dump(de,"DT.sav")

joblib.dump(ra, "RF.sav")