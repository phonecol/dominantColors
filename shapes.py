import turtle

class Polygon:
    def __init__(self,sides,name,size,color="black",line_thickness=2):
        self.sides = sides
        self.name = name
        self.size = size
        self.color = color
        self.line_thickness = line_thickness
        self.interior_angles = (self.sides-2)*180
        self.angle = self.interior_angles/self.sides

    def draw(self):
        turtle.color(self.color)
        turtle.pensize(self.line_thickness)
        for i in range(self.sides):
            turtle.forward(100)
            turtle.right(180-self.angle)


class Square(Polygon):
    def __init__(self,size,color="black",line_thickness=2):
        super().__init__(4,"Square", size, color, line_thickness)

    def draw(self):
        turtle.begin_fill()
        super().draw()
        turtle.end_fill()

square = Square(color="#123abc", size = 300)
print(square.draw())

turtle.done()