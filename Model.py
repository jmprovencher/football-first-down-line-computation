class Model:
    def __init__(self, modelImage):
        self.image = modelImage

        # if we want params like the size of the field, the pixels of the lines or ...
        self.params = dict([('upperLineY', 60), ('lowerLineY', 430)])
