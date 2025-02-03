class NestedInteger:
    def __init__(self, value=None):
        if value is None:
            self.value = []
        else:
            self.value = value

    def add(self, elem):
        # Ensure that self.value is a list for adding elements
        if not isinstance(self.value, list):
            self.value = [self.value]
        self.value.append(elem)

    def isInteger(self):
        return not isinstance(self.value, list)

    def getInteger(self):
        if self.isInteger():
            return self.value
        return None

    def getList(self):
        if not self.isInteger():
            return self.value
        return None

    def __iter__(self):
        if self.isInteger():
            raise TypeError("'NestedInteger' object is not iterable")
        # Ensures that self.value exists before iterating
        return iter(self.value if self.value else [])

def list_to_nested_integer(lst):
    if isinstance(lst, int):
        # Base case: If `lst` is an integer, return a NestedInteger holding that integer.
        return NestedInteger(lst)
    else:
        # Recursive case: `lst` is a list, create a NestedInteger to hold the list.
        ni = NestedInteger()
        for item in lst:
            # Recursively convert each item in the list to NestedInteger and add it.
            ni.add(list_to_nested_integer(item))
        return ni