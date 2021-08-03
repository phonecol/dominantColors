import matplotlib.pyplot as plt

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x,y)

    def plot(self):
        plt.scatter(self.x, self.y)

a = Point(3,4)
b = Point(2,2)
c = a+b

print(c.x, c.y)
