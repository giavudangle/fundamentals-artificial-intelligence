#Khai báo đồ thị
graph={
    'A':['B','C'],
    'B':['D','E'],
    'C':['E','I'],
    'D':['F'],
    'E':['F','J'],
    'F':['G','H'],
    'G':[], 'H':[],
    'I':['K'],
    'J':[],'K':[],
}

#Bài 1. Viết chương trình khai báo đồ thị và duyệt đồ thị theo giải thuật BreadthFirst Search (BFS) biết rằng BFS làm việc theo cơ chế Queue (hàng đợi) – FIFO
#(First In First Out). Sau đó xuất kết quả duyệt đồ thị ra màn hình.
#khai bao giai thuat BFS
def BFS(graph,start):
    queue=[start]
    visited=[]

    while queue: #queue khac rong thi chay tiep tuc
        node=queue.pop(0)
        if node in visited:
            continue
        visited.append(node)
        for nextNode in graph[node]:
            queue.append(nextNode)
    return visited
#call function BFS
print("Ket qua duyet do thi theo BFS: ")
print(BFS(graph,'A'))

#Bài 2. Dựa và giải thuật BFS. Sinh viên viết chương trình khai báo đồ thị và duyệt
#đồ thị theo giải thuật Depth-First Search (DFS) biết rằng DFS làm việc theo cơ chế
#Stack (ngăn xếp) – LIFO (Last In First Out). Sau đó xuất kết quả duyệt đồ thị ra màn hình.
def DFS(graph,start,visited):
    if start not in visited:
        visited.append(start)
    for n in graph[start]:
        DFS(graph,n,visited)
    return visited
visited=DFS(graph,'A',[])
print("Duyet do thi theo DFS: ")
print(visited)

#Bài 3. Viết chương trình cải tiến giải thuật Breadth-First Search (BFS) để in ra
#đường đi từ vị trí start đến vị trí end trong đồ thị.
def BFS_Path(graph,start,end):
    queue=[(start,[start])]
    while queue:
        node,path=queue.pop(0)
        for nextNode in graph[node]:
            if nextNode in path:
                continue
            elif nextNode==end:
                return path+[nextNode]
            else:
                queue.append((nextNode,path+[nextNode]))
#Call function BFS_Path
print("DUYET DO THI THEO BFS")
start=input("Nhap vi tri bat dau: ")
end=input("Nhap vi tri ket thuc: ")
print("Duong di tu "+start+" den "+end+": ")
print(BFS_Path(graph,start,end))
print("------------------------------------")

#Bài 4. Dựa vào Bài 3, sinh viên viết chương trình cải tiến giải thuật Depth-First
#Search (DFS) để in ra đường đi từ vị trí start đến vị trí end trong đồ thị.
def DFS_Path(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == goal:
                return path
            visited.add(vertex)
            for neighbor in graph[vertex]:
                stack.append((neighbor, path + [neighbor]))
#Call functionn DFS_Path
print("DUYET DO THI THEO DFS")
start=input("Nhap vi tri bat dau: ")
end=input("Nhap vi tri ket thuc: ")
print("Duong di tu "+start+" den "+end+": ")
print(DFS_Path(graph,start,end))