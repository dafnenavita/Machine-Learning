import math
import random

class Node:
    def __init__(self, instance, class_instance, attribute_val, left, right, label):
        self.instance = instance
        self.class_instance = class_instance
        self.attribute_val = attribute_val
        self.left = left
        self.right = right
        self.label = label



def getInstances(training_data):
    instance = []
    for index, row in training_data.iterrows():
        instance.append(index)
    return instance

def getClassLabels(training_data, class_label):
    class_instance = []
    for class_value in training_data[str(class_label)]:
        class_instance.append(class_value)
    return class_instance

def entropy(class_instance):
    pure = 0
    nonpure = 0
    total_length = len(class_instance)
    # entropy_value = 0
    for value in class_instance[:]:
        if value > 0:
            pure += 1
        else:
            nonpure += 1

    if pure > 0:
        entropy_pure = -1 * (pure / total_length) * math.log((pure / total_length), 2)
    else:
        entropy_pure = 0
    if nonpure > 0:
        entropy_nonpure = -1 * (nonpure / total_length) * math.log((nonpure / total_length), 2)
    else:
        entropy_nonpure = 0
    entropy_value = (entropy_pure + entropy_nonpure)
    return entropy_value


def getChild(data, attribute_val_value, instance, class_instance):
    left_child = []
    left_child_classes = []
    right_child = []
    right_child_classes = []
    attribute_val_instance = []
    for value in data[str(attribute_val_value)]:
        attribute_val_instance.append(value)

    for (instance, class_value) in zip(instance, class_instance):
        if attribute_val_instance[instance] > 0:
            right_child.append(instance)
            right_child_classes.append(class_value)
        else:
            left_child.append(instance)
            left_child_classes.append(class_value)
    left = [left_child, left_child_classes]
    right = [right_child, right_child_classes]
    return [left, right]


def accuracy(node, data):
    p = 0
    temp = Node(None, None, None, None, None, None)
    instance = []
    for index, row in data.iterrows():
        instance.append(index)
    col_val = data.columns.values
    class_label = col_val[-1]
    class_instance = []
    for class_value in data[str(class_label)]:
        class_instance.append(class_value)
    for i in range(0, len(instance)):
        temp = node
        while not (check_pure(temp)):
            if data[temp.attribute_val][i] > 0:
                temp = temp.right
            else:
                temp = temp.left
        if temp.attribute_val == class_instance[i]:
            p += 1
    return p / len(instance)

def tempAccr(node, data, label):
    p = 0
    temp = Node(None, None, None, None, None, None)
    instance = []
    for index, row in data.iterrows():
        instance.append(index)
    col_val = data.columns.values
    class_label = col_val[-1]
    class_instance = []
    for class_value in data[str(class_label)]:
        class_instance.append(class_value)
    for i in range(0, len(instance)):
        temp = node
        while not (check_pure(temp)):
            if temp.label == label:
                break
            else:
                if data[temp.attribute_val][i] > 0:
                    temp = temp.right
                else:
                    temp = temp.left
        if temp.label == label:
            avgValue = getAvgClass(temp.class_instance)
            if (class_instance[i] and avgValue >= 0.5) or (not (class_instance[i]) and avgValue < 0.5):
                p += 1
        else:
            if temp.attribute_val == class_instance[i]:
                p += 1
    return p / len(instance)


def postPruneAccr(node, data, label):
    p = 0
    temp = Node(None, None, None, None, None, None)
    instance = []
    for index, row in data.iterrows():
        instance.append(index)
    col_val = data.columns.values
    class_label = col_val[-1]
    class_instance = []
    for class_value in data[str(class_label)]:
        class_instance.append(class_value)
    for i in range(0, len(instance)):
        temp = node
        while not (check_pure(temp)):
            if temp.label in label:
                break
            else:
                if data[temp.attribute_val][i] > 0:
                    temp = temp.right
                else:
                    temp = temp.left
        if temp.label in label:
            avgValue = getAvgClass(temp.class_instance)
            if (class_instance[i] and avgValue >= 0.5) or (not (class_instance[i]) and avgValue < 0.5):
                p += 1
        else:
            if temp.attribute_val == class_instance[i]:
                p += 1
    return (p / len(instance))

def getAvgClass(class_instance):
    if len(class_instance) > 0:
        total = 0
        for class_value in class_instance[:]:
            total += class_value
        return total / len(class_instance)
    else:
        return random.randint(0, 1)

def check_pure(node):
    root_node = Node(None, None, None, None, None, None)
    root_node = node
    return type(root_node.attribute_val) is int


def count_total_nodes(node):
    if check_pure(node):
        return 1
    else:
        left_total = count_total_nodes(node.left)
        right_total = count_total_nodes(node.right)
        return left_total + right_total + 1


def count_pnodes(node):
    if check_pure(node):
        return 1
    else:
        left_count = count_pnodes(node.left)
        right_count = count_pnodes(node.right)
        return left_count + right_count



def count_npnodes(node):
    if check_pure(node):
        return 0
    else:
        return count_npnodes(node.left) + count_npnodes(node.right) + 1

def count_pruned_nodes(node, label):
    if check_pure(node) or node.label in label:
        return 1
    else:
        left_prune = count_pruned_nodes(node.left, label)
        right_prune = count_pruned_nodes(node.right, label)
        return left_prune + right_prune + 1


def count_after_prune(node, label):
    if check_pure(node) or node.label in label:
        return 1
    else:
        left_after_prune = count_after_prune(node.left, label)
        right_after_prune = count_after_prune(node.right, label)
        return left_after_prune + right_after_prune


nodes_values = []
pruneAccuracy = []

def getprune_nodes(node, prune_nodes_num, accuracy, root_node, data):
    if not (check_pure(node)):
        if check_pure(node.left) and check_pure(node.right):
            tempAccuracy = tempAccr(root_node, data, node.label)
            if accuracy < tempAccuracy:
                if len(nodes_values) < prune_nodes_num:
                    nodes_values.append(node.label)
                    pruneAccuracy.append(tempAccuracy)
                else:
                    if min(pruneAccuracy) < tempAccuracy:
                        for i in range(0, prune_nodes_num):
                            if pruneAccuracy[i] == min(pruneAccuracy):
                                pruneAccuracy[i] = tempAccuracy
                                nodes_values[i] = node.label
        else:
            getprune_nodes(node.left, prune_nodes_num, accuracy, root_node, data)
            getprune_nodes(node.right, prune_nodes_num, accuracy, root_node, data)




def build_decsiontree(instance, class_instance, attribute_val, data, label):
    class_average = float(getAvgClass(class_instance))
    length_attr = len(attribute_val)
    if 0 < class_average < 1 and length_attr > 0:
        max_attribute_val = ''
        max_info_gain = -999
        child_inst = []
        for attribute_val_value in attribute_val[:]:
            root_node_entropy = entropy(class_instance)
            child = getChild(data, attribute_val_value, instance, class_instance)
            left_depth = len(child[0][0])
            right_depth = len(child[1][0])
            root_depth = len(instance)
            info_gain = root_node_entropy - (
                        ((left_depth / root_depth) * entropy(child[0][1])) + ((right_depth / root_depth) *
                                                                                    entropy(child[1][1])))
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                max_attribute_val = attribute_val_value
                child_inst = child

        root_node = Node(None, None, None, None, None, None)
        root_node.instance = instance
        root_node.class_instance = class_instance
        root_node.attribute_val = max_attribute_val
        attribute_val = list(attribute_val)
        attribute_val.remove(max_attribute_val)
        ch1 = child_inst[0][0]
        ch2 = child_inst[0][1]
        ch3 = child_inst[1][0]
        ch4 = child_inst[1][1]
        root_node.left = build_decsiontree(ch1, ch2, attribute_val, data, (label * 2))
        root_node.right = build_decsiontree(ch3, ch4, attribute_val, data, ((label * 2) + 1))
        root_node.label = label
        return root_node
    else:
        root_node = Node(None, None, None, None, None, None)
        root_node.instance = instance
        root_node.class_instance = class_instance
        root_node.label = label
        if len(attribute_val) == 0:
            if class_average >= 0.5:
                root_node.attribute_val = 1
            else:
                root_node.attribute_val = 0
        else:
            root_node.attribute_val = int(class_average)
        return root_node
    return 0


def print_tree(node, indent):
    root_node = Node(None, None, None, None, None, None)
    root_node = node
    print("| " * indent, end="")
    if check_pure(root_node.left):
        print(root_node.attribute_val, ' = 0 : ', root_node.left.attribute_val)
    else:
        print(root_node.attribute_val, ' = 0 : ')
        print_tree(root_node.left, indent + 1)
    print("| " * indent, end="")
    if check_pure(root_node.right):
        print(root_node.attribute_val, ' = 1 : ', root_node.right.attribute_val)
    else:
        print(root_node.attribute_val, ' = 1 : ')
        print_tree(root_node.right, indent + 1)





