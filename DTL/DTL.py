"""
CPS4902 Machine Learning ---- Decision Tree Learning
Junyan Dai
1025584
Instructor: Dr. Jenny Li
10/7/2018
"""

import numpy as np
import pandas as pd
###########################################################################################################
"""
Create Tree class
"""
class Tree(object):
    """The basic node of tree structure"""

    def __init__(self, name):
        super(Tree, self).__init__()
        self.name = name
        self.parent = None
        self.child = {}
        self.value = name

    def __repr__(self):
        return 'TreeNode(%s)' % self.name

    def __contains__(self, item):
        return item in self.child

    def __len__(self):
        """return number of children node"""
        return len(self.child)

    def __bool__(self, item):
        """always return True for exist node"""
        return True

    @property
    def path(self):
        """return path string (from root to current node)"""
        if self.parent:
            return '%s %s' % (self.parent.path.strip(), self.name)
        else:
            return self.name

    # def get_child(self, name, defval=None):
    #     """get a child node of current node"""
    #     return self.child.get(name, defval)

    def add_child(self, name, obj):
        """add a child node to current node"""
        # if obj and not isinstance(obj, Tree):
        #     raise ValueError('TreeNode only add another TreeNode obj as child')
        # if obj is None:
        #     obj = Tree(name)
        obj.parent = self
        self.child[name] = obj
        return obj

    def del_child(self, name):
        """remove a child node from current node"""
        if name in self.child:
            del self.child[name]

    # def find_child(self, path, create=False):
    #     """find child node by path/name, return None if not found"""
    #     # convert path to a list if input is a string
    #     path = path if isinstance(path, list) else path.split()
    #     cur = self
    #     for sub in path:
    #         # search
    #         obj = cur.get_child(sub)
    #         if obj is None and create:
    #             # create new node if need
    #             obj = cur.add_child(sub)
    #         # check if search done
    #         if obj is None:
    #             break
    #         cur = obj
    #     return obj

    def items(self):
        return self.child.items()

    def dump(self, indent=0):
        """dump tree to string"""
        tab = '    '*(indent-1) + ' |- ' if indent > 0 else ''
        print('%s%s%s%s' % (tab, self.value, "  ", self.name))
        for name, obj in self.items():
            obj.dump(indent+1)
# tree modified from xuelians' CSDN Blog，https://blog.csdn.net/xuelians/article/details/79999284?utm_source=copy

    def set_value(self, value):
        self.value = value


###########################################################################################################


# trainning data for DTL
attributes_data = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est']
examples_data = pd.DataFrame(data=[['T','F','F','T','Some','$$$','F','T','French','0-10','T'],
                             ['T','F','F','T','Full','$','F','F','Thai','30-60','F'],
                             ['F','T','F','F','Some','$','F','F','Burger','0-10','T'],
                             ['T','F','T','T','Full','$','F','F','Thai','10-30','T'],
                             ['T','F','T','F','Full','$$$','F','T','French','>60','F'],
                             ['F','T','F','T','Some','$$','T','T','Italian','0-10','T'],
                             ['F','T','F','F','None','$','T','F','Burger','0-10','F'],
                             ['F','F','F','T','Some','$$','T','T','Thai','0-10','T'],
                             ['F','T','T','F','Full','$','T','F','Burger','>60','F'],
                             ['T','T','T','T','Full','$$$','F','T','Italian','10-30','F'],
                             ['F','F','F','F','None','$','F','F','Thai','0-10','F'],
                             ['T','T','T','T','Full','$','F','F','Burger','30-60','T']],
                        columns = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est','Target'])
###########################################################################################################
"""
Implement a method for calculating Entropy of a factor
"""
def get_entropy(attribute):
    #Calculate the entropy of particular attribute.
    elements, counts = np.unique(attribute, return_counts=True)
    entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
                      for i in range(len(elements))])
    return entropy
###########################################################################################################
"""
Implement a method for calculating Remainder of a factor
"""
def get_remainder(examples, attribute):
    # Calculate the elements and the corresponding counts for the split attribute
    elements, counts = np.unique(examples[attribute], return_counts=True)

    # Calculate the remainder
    remainder = np.sum(
        [(counts[i] / np.sum(counts)) * get_entropy(examples.where(examples[attribute] == elements[i]).dropna()["Target"])
         for i in range(len(elements))])
    return remainder
###########################################################################################################
"""
Implement a method for calculating Gain of a factor
"""
def infogain(examples, attribute):
    # Calculate the Total Entropy
    total_entropy = get_entropy(examples["Target"])

    # Calculate the Information Gain: Total Entropy - remainder
    information_gain = total_entropy - get_remainder(examples, attribute)
    return information_gain

###########################################################################################################
"""
Decision Learning Tree(DTL) Function
"""
def DTL(examples, attributes,default = None):
    elements = np.unique(examples["Target"])
    if len(examples) == 0:
        return default
    elif len(elements) == 1:
        return Tree(elements[0])
    elif attributes is None:
        return None
    else:
        best = ChooseAttribute(attributes, examples)
        tree = Tree(best)
        elements, counts = np.unique(examples[best], return_counts=True)
        tempattri = attributes
        tempattri.remove(best)
        for i in range(len(elements)):
            examples_i = examples.loc[examples[best] == elements[i]]
            subtree = DTL(examples_i, tempattri,mode(examples))
            if subtree is not None:
                subtree.set_value(elements[i])
                tree.add_child(elements[i], subtree)
        return tree


###########################################################################################################
"""
Choose the best attribute
"""
def ChooseAttribute(attributes, examples):
    largestIG = 0
    index = 0
    # Compare all attributes' Information Gain expect Target
    for i in range(len(attributes)):
        information_gain = infogain(examples, attributes[i])
        if information_gain > largestIG:
            largestIG = information_gain
            index = i
    # Find the best attribute, which means find the largest Information Gain
    best = attributes[index]
    return best

"""
Mode Funtion
"""
def mode(examples):
    elements, counts = np.unique(examples["Target"], return_counts=True)
    largest = 0
    index = 0
    for i in range(len(elements)):
        if counts[i]>largest:
            largest = counts[i]
            index = i
    return Tree(elements[index])

"""
Main
"""
if __name__ == '__main__':
    decision_tree = DTL(examples_data, attributes_data) #implement DTL Function
    decision_tree.dump() #print the tree
