# Khai báo Graph
class Graph:
    # constructor
    def __init__(self, adjacency_list, H):
        self.adjacency_list = adjacency_list
        self.H = H

    # child of vertex v
    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function (h) is node value
    def h(self, n):
        return self.H[n]

    # A* algorithm
    def a_star_algorithm(self, start_node, end_node):
        # open_list : node list have been visited BUT their neighbors HAVEN'T been visited
        # closed_list : node list have been visited AND their neighbors HAVE been visited
        open_list = set([start_node])
        closed_list = set([])

        # g is distance from node x -> node y
        # list g store distance from x to connected nodes
        g = {}
        g[start_node] = 0

        # parents store parent of node
        parents = {}
        parents[start_node] = start_node

        while open_list:
            n = None

            # Find node with value is min cost of function f()
            # f = g + h

            for v in open_list:
                # if f{v} < f{n}
                if n == None or (g[v] + self.h(v) < g[n] + self.h(n)):
                    n = v  # Compare future cost and re-assign with min cost
            if n == None:
                print("Cannot find path")
                return None

            # If current node is end_node

            if n == end_node:
                # Store path
                path = []
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                # Add start_node to path
                path.append(start_node)
                # Reverse
                path.reverse()
                print("Path : {}".format(path))
                return path
            # If current not is not end_node
            # Create loop to update value of neighbors
            # Update h,g,f for nodes

            for (m, cost) in self.get_neighbors(n):
                # if current is not exist in open_list and closed_list
                # Add to open_list,update g and parent of node
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                # Else if we find another shorter path
                # We update value,parent of node
                # If node has been visited, we store it in closed_list
                # But - if current node is in another shorter path of hasn't visited
                # We can plug out current node and push to open_list
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print("Doesnt exist path")
        return None


# --------------------------------------------

# adjacency_list = {
#     'A': [('C', 9), ('D', 7), ('E', 13), ('F', 20)],
#     'C': [('H', 6)],
#     'D': [('E', 4), ('H', 8)],
#     'E': [('K', 4), ('I', 3)],
#     'F': [('I', 6), ('G', 4)],
#     'H': [('K', 5)],
#     'K': [('B', 6)],
#     'I': [('K', 9), ('B', 5)],
# }
#
# heuristic = {
#     'A': 14,
#     'B': 0,
#     'C': 15,
#     'D': 6,
#     'E': 8,
#     'F': 7,
#     'G': 12,
#     'H': 10,
#     'K': 2,
#     'I': 4
# }


adjacency_list = {
    'A': [('T', 118), ('S', 140), ('Z', 75)],
    'B': [('F', 211), ('G', 66), ('P', 100)],
    'C': [('D', 120), ('R', 146)],
    'D': [('C', 120), ('M', 75)],
    'F': [('B', 211), ('S', 99)],
    'G': [('B', 66)],
    'L': [('M', 70), ('T', 111)],
    'M': [('D', 75), ('L', 70)],
    'O': [('S', 151), ('Z', 71)],
    'P': [('B', 100), ('R', 97)],
    'R': [('C', 146), ('P', 97), ('S', 80)],
    'S': [('A', 140), ('F', 99), ('O',151),('R', 60)],
    'T': [('A', 118), ('L', 111)],
    'Z': [('A', 75), ('O', 71)],


}

heuristic = {
    'A': 366,
    'B': 0,
    'C': 160,
    'D': 242,
    'F': 178,
    'G': 77,
    'L': 244,
    'M': 241,
    'O': 380,
    'P': 98,
    'R': 193,
    'S': 253,
    'T': 329,
    'Z': 374,
}

my_graph = Graph(adjacency_list, heuristic)
my_graph.a_star_algorithm('A', 'B')
