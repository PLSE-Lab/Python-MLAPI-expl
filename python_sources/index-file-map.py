class IndexFileMap:
    def __init__(self):
        self.start_index_list = []
        self.path_list = []
        self.previous_path = None

    def add(self, start_index, path):
        """add files with increasing index"""
        if self.previous_path is not path:
            self.start_index_list.append(start_index)
            self.path_list.append(path)
            self.previous_path = path

    def get(self, paper_index):
        list_index = self.__bin_search(paper_index, 0, len(self.start_index_list) - 1)
        if list_index is -1:
            return "not found"
        else:
            return self.start_index_list[list_index], self.path_list[list_index], list_index

    def save(self, filename):
        file = open(filename, "w")
        for i in range(len(self.start_index_list)):
            file.write('%s\n' % str(self.start_index_list[i]))
            file.write('%s\n' % str(self.path_list[i]))

    def load(self, filename):
        self.start_index_list = []
        self.path_list = []

        file = open(filename, "r")
        iterator = iter(file)

        start_index = next(iterator, None)
        path = next(iterator, None)
        while path is not None:
            # [:-1] to remove linebreak
            self.start_index_list.append(int(start_index[:-1]))
            self.path_list.append(path[:-1])

            start_index = next(iterator, None)
            path = next(iterator, None)

    def __bin_search(self, paper_index, l, r):

        def check_match(i):
            if i >= len(self.start_index_list) - 1:
                return True
            elif self.start_index_list[i] <= paper_index < self.start_index_list[i + 1]:
                return True
            return False

        if l > r:
            return -1

        mid = l + (r - l) // 2

        if check_match(mid):
            return mid

        elif paper_index < self.start_index_list[mid]:
            return self.__bin_search(paper_index, l, mid - 1)
        else:
            return self.__bin_search(paper_index, mid + 1, r)
