#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 1. Viết một chương trình Python in ra dãy số Fibonacci giữa 0 và 50.
def fibo_find():
    n1, n2 = 0, 1
    for i in range(0, 50):
        print(n1)
        n = n1 + n2
        n1 = n2
        n2 = n

fibo_find()


# In[4]:


#2. Viết một chương trình Python đếm số các số chẵn và số các số lẻ của một dãy số

def is_even(n: int) -> bool:
    return True if n % 2 == 0 else False


def count_number():
    count_odd = 0
    count_even = 0
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(arr)-1):
        if is_even(arr[i]):
            count_even += 1
        else:
            count_odd += 1
    print("Odd -> " + repr(count_odd))
    print("Even -> " + repr(count_even))




count_number()


# In[5]:


"""
3. Viết một chương trình Python xây dựng cấu trúc sau, bằng việc sử dụng vòng lặp for
lồng nhau.
*
* *
* * *
* * * *
* * * * *
"""

def print_star():
    for i in range(0,6):
        for j in range(i):
            print("* ",end=" ")
        print(" ")


print_star()


# In[6]:


#4. Viết một chương trình Python tính tổng tất cả các số trong một danh sách. 

def sum_list(list: []) -> int :
    sum = 0
    for i in range(len(list)):
        sum += list[i]
    return sum


test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(sum_list(test))


# In[ ]:




