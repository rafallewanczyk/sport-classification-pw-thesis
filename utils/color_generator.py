import hashlib


class ColorGenerator:
    def __init__(self):
        self.generated_colors_memory = {}

    def id_to_random_color(self, any, normalized=False):
        number = str(any)
        if not number in self.generated_colors_memory:
            numByte = str.encode(number)
            hashObj = hashlib.sha1(numByte).digest()
            r, g, b = hashObj[-1], hashObj[-2], hashObj[-3]
            if normalized:
                r = r / 255
                g = g / 255
                b = b / 255
            self.generated_colors_memory[number] = (r, g, b)
            return r, g, b
        else:
            return self.generated_colors_memory[number]
