#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Nhập và đọc dữ liệu
data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
airline_tweets.head()

#Phân tích dữ liệu
plot_size = plt.rcParams["figure.figsize"]
print(plot_size[0])
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
#Phân tích tình cảnh
airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%',
                                                    colors=["red", "yellow","green"])
#Phân bố tình cảnh với từng hãng hàng không
airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')

#Thư viện Seaborn
import seaborn as sns

sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)

#Làm sạch dữ liệu
features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
processed_features = []

for sentence in range(0, len(features)):
    #Xóa tất cả các kí tự đặt biệt
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    
    #Xóa tất cả các kí tự đơn
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    
    #Xóa kí tự đơn từ lúc bắt đầu
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    
    #Thay thế nhiều khoảng trắng bằng một khoảng trắng
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags= re.I)
    
    #Xóa kí tự 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    
    #Chuyển đổi sang chữ thường
    processed_feature = processed_feature.lower()
    
    processed_features.append(processed_feature)


#Bag of Words

#Chuyển đổi văn bản ở dạng số
 #Vocab = [I, love, to, be, cared, for, by, my ,lover, Ty, hihi]
 #[1,1,1,1,1,0,0,0,0,0,0,0,0]


#TF-IDF
 #(TF-IDF là sự kết hợp của hai thuật ngữ. Tần suất thuật ngữ và
 #tần suất Tài liệu nghịch đảo)
 #TF = (Frequency of a word in the document)/(Total words in the document)
 #IDF = Log((Total number of docs)/(Number of docs containing the word))


#TF-IDF sử dụng thư viện SCIKIT-LEARN
#from nltk.corpus import stopwords
#from sklearn,feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer (max_features=2500, min_df=0.8, stop_words=stopwords.words('Endlish'))
#processed_features = vectorizer.fit_transfrom(processed_features).toarray()


#Chia dữ liệu thành các tập huấn luyện và kiểm tra
#train_test_split lớp từ sklearn.model_selection mô đun để chia sẻ dữ liệu.
#phường thúc này nhận tập tập hợp tính năng làm tham số đầu tiên.
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


#Đào tạo người mẫu
#from sklern.ensemble import RandomForestClassifier

#text_classifier = RandomForrestClassifier(n_estimators=200, random_state=0)
#text_classifier.fit(X_train,Y_train)

#Đưa ra dự đoán và đánh giá
#predictions = text_classifier.predict(X_test)

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#print(accuracy_score(y_test, predictions))


# In[ ]:





# In[ ]:
