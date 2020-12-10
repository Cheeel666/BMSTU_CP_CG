from PIL import Image
import numpy as np
from math import sqrt, exp
#python3 -m black main.py 

def distance_between(a, b):
    
    
    return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)


def decay(x, dispersion):
    return exp(-x / dispersion)

def distance_over_windows(data, x,y,x1,y1,radius):
    dispersion = 15000
    accum = 0
    for i in range (-radius, radius + 1):
        for j in range(-radius, radius + 1):
            accum += distance_between(data[x][y], data[x1][y1])
    return decay(accum, dispersion)


class ImageFilter:
    def __init__(self, name):
        self.name = name
        self.img = Image.open(name).convert("RGB")
        self.img_pixels = np.array(Image.open(name).convert("RGB"))
        self.img_copy = Image.fromarray(np.asarray(self.img))

    def save_result(self):
        self.img_copy = Image.fromarray(self.img_pixels)
        self.img_copy.save("result.png")

    def use_filter(self):
        width, height = self.img.size
        window_radius = 3 # 2 is enough for the text
        data = []
        for x in range(window_radius + 1,height - window_radius - 1):
            for y in range(window_radius,width - window_radius - 1):
                weight_map = [[0 for x in range(width)] for y in range(height)]
                weight_data = 0
                norm = 0
                for x1 in range(window_radius + 1,height - window_radius - 1):
                    for y1 in range(window_radius + 1, width - window_radius - 1):
                        
                        
                        weight = distance_over_windows(self.img_pixels,x,y,x1,y1,window_radius)
                        norm += weight  
                        weight_map[x1][y1] = weight
            print(x1)
        print(weight_map)

if __name__ == "__main__":
    myImg = ImageFilter("b.png")
    myImg.use_filter()
    myImg.save_result()
#604 606 3
#244 174 3