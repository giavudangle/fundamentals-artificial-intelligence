import numpy as np
import datetime


# 1. Viết hàm trong chương trình Python in ra dãy số Fibonacci giữa 0 và 50 (sử dụng hàm đệ quy)
def print_fibonacci(n):
    if n <= 1:
        return n
    return print_fibonacci(n - 1) + print_fibonacci(n - 2)


def run_fibo(n):
    for i in range(n):
        print(print_fibonacci(i))


# run_fibo(50)


# 2. Viết một hàm Python tính tổng tất cả các số trong một danh sách

def sum_list(list_array: []) -> int:
    sum_all = 0
    for i in range(len(list_array)):
        sum_all += list_array[i]
    return sum_all


arr = [6, 9, 1, 5, 2, 6]


# print(sum_list(arr))

# 3. Viết một hàm Python đầu vào là một số và
# kiểm tra xem số đó có phải là số nguyên tố hay
# không

def is_prime(n: int) -> bool:
    count_div = 1
    for i in range(1, n):
        if n % i == 0:
            count_div += 1
    if count_div > 2:
        return False
    return True


def check_prime(n: int):
    if is_prime(n):
        print("Is prime")
    else:
        print("Not prime")


# check_prime(12)

# 4. Khởi tạo 2 ma trận 3x3, áp dụng các phép toán cộng,
# trừ 2 ma trận và xuất ra ma trận kết quả.

def create_matrix():
    a = np.arange(2, 11).reshape(3, 3)
    return a


def sum_matrix(a, b):
    return a + b


def minus_matrix(a, b):
    return a - b


matrix_one = create_matrix()
matrix_two = create_matrix()


# res_sum = sum_matrix(matrix_one,matrix_two)
# res_minus = minus_matrix(matrix_one,matrix_two)
#
# print(res_sum)
# print(res_minus)

# 5. Xây dựng lớp KhachHang

class KhachHang:
    client_id = ""
    client_name = ""
    year = 0
    address = ""
    phone = ""
    money = 0

    def __init__(self):
        self.client_id = ""

    def NhapThongTin(self):
        print("Enter client_id : ", end=" ")
        self.client_id = input()
        print("Enter client_name: ", end=" ")
        self.client_name = input()
        print("Enter year : ", end=" ")
        self.year = int(input())
        print("Enter address : ", end=" ")
        self.address = input()
        print("Enter phone : ", end=" ")
        self.phone = input()
        print("Enter money : ", end=" ")
        self.money = float(input())

    def XuatThongTin(self):
        print(" client_id : " + repr(self.client_id))
        print(" clienclient_namet_id : " + repr(self.client_name))
        print(" year : " + repr(self.year))
        print(" address : " + repr(self.address))
        print(" phone : " + repr(self.phone))
        print(" money : " + repr(self.money))
        print(" age : " + repr(self.get_client_age()))

    def get_client_age(self):
        now = datetime.datetime.now()
        return now.year - self.year


#
# client = KhachHang()
# client.NhapThongTin()
# client.XuatThongTin()

# BÀI 6 : Xây dựng lớp giảng viên sinh viên
class Person:
    role = ""
    name = ""
    id = ""

    def __init__(self):
        self.id = ""

    def Input(self):
        self.id = input('Enter id : ')
        self.role = input('Enter role : ')
        self.name = input('Enter name : ')

    def Output(self):
        print(' id : ' + repr(self.id))
        print(' name : ' + repr(self.name))
        print(' role : ' + repr(self.role))

    def calculate_award(self):
        print("PERSON AWARDS")


class Student(Person):
    average_mark = 0

    def __init__(self):
        self.average_mark = 0

    def Input(self):
        super(Student, self).Input()
        self.average_mark = int(input("Enter student average mark : "))

    def Output(self):
        print("STUDENT")
        super(Student, self).Output()
        print("Mark " + repr(self.average_mark))

    def calculate_award(self):
        super().calculate_award()
        if self.average_mark <= 8:
            return "Average mark must be greater than 8"
        else:
            return "Congratulation STUDENT ^^"

    def get_award(self):
        return self.calculate_award()


class Lecturer(Person):
    science_count = 0

    def __init__(self):
        self.science_count = 0

    def Input(self):
        super().Input()
        self.science_count = int(input("Enter lecturer science_count : "))

    def Output(self):
        super().calculate_award()
        print("LECTURER")
        print("Science_Count " + repr(self.science_count))

    def calculate_award(self):
        if self.science_count < 2:
            return "Science count must be greater than 2"
        else:
            return "Congratulation LECTURER ^^"

    def get_award(self):
        return self.calculate_award()


#
# s = Student()
# s.Input()
# s.Output()
# print(s.get_award())

# l = Lecturer()
# l.Input()
# l.Output()
# print(l.get_award())


class Shape:
    width = 0
    height = 0

    def __init__(self):
        self.width = 0
        self.height = 0

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width


class Triangle(Shape):

    def __init__(self):
        self.height = super().get_height()
        self.base = super().get_width()

    def input(self):
        print("Enter value of triangle")
        self.height = int(input("Height : "))
        self.base = int(input("Base: "))

    def output(self):
        print("Triangle ")
        print("Base " + repr(self.base))
        print("Height " + repr(self.height))

    def area(self) -> float:
        return (self.height * self.base) / 2

    def get_class_name(self):
        return "Triangle"


class Rectangle(Shape):

    def __init__(self):
        self.height = super().get_height()
        self.width = super().get_width()

    def input(self):
        print("Enter value of rectangle")
        self.height = int(input("Height : "))
        self.width = int(input("Width: "))

    def output(self):
        print("Rectangle ")
        print("Width " + repr(self.width))
        print("Height " + repr(self.height))

    def area(self):
        return self.height * self.width

    def get_class_name(self):
        return "Rectangle"


def my_input(n):
    result = []
    for i in range(0, n ):
        print("Enter shape => " + repr(i))
        print("Enter shape type :")
        print(" Triangle is T")
        print(" Rectangle is C")
        choice = input()
        if choice != 'C' and choice != 'T':
            print("Invalid")
            return
        if choice == 'C':
            r = Rectangle()
            r.input()
            result.append(r)
        if choice == 'T':
            t = Triangle()
            t.input()
            result.append(t)
    return result

def my_output(list):
    sum_area = 0
    for i in range(0,len(list)):
        list[i].output()
        sum_area += list[i].area()
        print("Area of " + list[i].get_class_name() + " " + repr(list[i].area()))
    print("Sum area is " + repr(sum_area))


def my_main():
    n = int(input("Enter number of Shape : "))
    temp = my_input(n)
    print("\n")
    my_output(temp)


my_main()
