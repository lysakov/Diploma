import csv

class MatrixStorage(object):

    def __init__(self, file_path : str, matrix_factory=None):
        self.path = file_path
        self.matrix_factory = matrix_factory
        self.field_names = ["Matrix", "Vector", "Label"]

    def write(self, matrix, vector, label=None):
        with open(self.path, "a") as storage:
            csv_writer = csv.DictWriter(storage, self.field_names)
            csv_writer.writerow({"Matrix": str(matrix), "Vector": str(vector), "Label": label})

    def read(self, ind, field : str):
        if self.matrix_factory is None:
            raise ValueError("Impossible return matrix without matrix factory")

        with open(self.path, "r") as storage:
            csv_reader = csv.DictReader(storage, self.field_names)
            
            return self.matrix_factory.fromString(
                next((x for i, x in enumerate(csv_reader) if i == ind), None)[field])

    def clean(self):
        with open(self.path, "w"):
            pass