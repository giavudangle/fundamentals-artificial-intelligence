import pandas as pd
import  numpy as np

#đọc dữ liệu từ file
file_path = 'D:/Test/SMSSpamCollection.txt'
df = pd.read_csv(file_path, delimiter='\t', header=None, skipinitialspace=True, names=['label', 'msg'])
#print(df)

#chuyển Label thành giá trị nhị phân 0: ham; 1: spam
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

#chia tập dữ liệu thành 2 tập: training data và test data theo tỉ lệ 7: 3
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,1], df.iloc[:,0], test_size=0.3, random_state=50)
print (X_train.head(10))
print (Y_train.head(10))


from sklearn.feature_extraction.text import CountVectorizer
# Khởi tạo vectorizer
vect = CountVectorizer()

#đếm các từ trong tập dữ liệu train và chuyển thành ma trận từ
#(Learn vocabulary and create document-term matrix in a single step)
train_dtm = vect.fit_transform(X_train)
train_dtm

#Chuyển tập dữ liệu text thành ma trận từ (transform testing data into a document-term matrix)
test_dtm = vect.transform(X_test)
test_dtm

#Chuyển tập traint_dtm thành array
train_arr = train_dtm.toarray()
train_arr

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, Y_train)

#DỰ đoán dữ liệu trong test(make predictions on test data using test_dtm
Y_pred = nb.predict(test_dtm)
Y_pred

#So sánh độ chính xác với các Label trong test có sẵn để biết bộ chính xác phân loại
#(compare predictions to true Labels)
from sklearn import metrics
print('accuracy= ', metrics.accuracy_score(Y_test, Y_pred))