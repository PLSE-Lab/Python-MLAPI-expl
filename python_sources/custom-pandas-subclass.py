import pandas as pd


class Pandas_Subclass(pd.DataFrame):
    """
    A way to create a Pandas subclass which initializes from another Pandas object without passing it into the class constructor.
    Allows complete overwriting of parent class constructor.
    Allows custom methods to be added onto Pandas objects (which can be created within the constructer itself).
    Ie. pass in a file_path to the class constructor which then calls pd.read_csv within __init__ which then assigns the returned DataFrame to self.
    Params:
        file_path (str): file_path passed into pd.read_csv().
    """

    def __init__(self, file_path):
        super().__init__(pd.read_csv(file_path))  # initialize subclass from DataFrame instance
        # self.__dict__.update(pd.read_csv(file_path).__dict__)  # the unpythonic way to do it

    def custom_method(self):
        print(self)  # returns .csv as Dataframe
        print(type(self))  # returns <class '__main__.Pandas_Subclass'>


if __name__ == '__main__':
    df = Pandas_Subclass('../input/winemag-data_first150k.csv')
    df.custom_method()