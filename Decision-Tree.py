__author__ = 'HANMANT LOKARE'

import pandas as pd
import HW_05_Lokare_Hanmant_Classifer
attributes = ["FlourOrOats", "Milk", "Sugar", "Butter or Margarine", "Egg",
              "Baking Powder", "Vanilla", "Salt", "Baking Soda", "Cream of tartar", "cinnamon",
              "allspice", "nutmeg", "ginger", "Canned Pumpkin_or_Fruit", "Apple Pie Spice",
              "ChocChips", "ChoppedWalnuts", "BrownSugarOrHoney", "Chopped Pears or Fruit", "Vegetable Oil",
              "Water", "Powdered Sugar", "Yogurt", "NUTS", "Type"]

"""
        This class will form the questions based on each values of attributes
        """
class If_question:
    """
            Construct a If_question instance.
            :param attribute: column number of attribute
            :param attribute_value: value of the attribute
            """

    def __init__(self, attribute, attribute_value):
        self.column_no = attribute
        self.value = attribute_value

    # Form the questions
    def best(self, example):
        val = example[self.column_no]
        if isinstance(val, int) or isinstance(val, float):
            return val >= self.value

    # String representation of the question
    def __repr__(self):
        return "If " + attributes[self.column_no] + " >= " + str(self.value) + ":"


"""
        This class represent the leaf node of the decision tree
        """
class final_node:

    def __init__(self, rows):
        self.final_value = type_count(rows)


"""
        This class represent the current node of the tree
        """
class current_node:
    """
             Current node instance.
             :param question: best question asked at this node
             :param right_tree: right sub-tree of this node
             :param left_tree: left sub-tree of this node
             """

    def __init__(self, question, right_tree, left_tree):
        self.question = question
        self.right_tree = right_tree
        self.left_tree = left_tree

"""
        This function will counts the number of observations
        """
def type_count(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


"""
        This function will divide the set bases on question asked
        """
def division_of_set(observations, question):
    left_part = []
    right_part = []
    for row in observations:
        if question.best(row):
            left_part.append(row)
        else:
            right_part.append(row)
    return left_part, right_part

"""
    This will find the gini_index of all observation
    """

def gini_index(observation_no):
    counts = type_count(observation_no)
    gini_index_value = 1
    for value in counts:
        prob_of_observation = counts[value] / float(len(observation_no))
        gini_index_value -= prob_of_observation ** 2
    return gini_index_value

"""
        This function will calculate the information gain of the given nodes
        """

def information_gain(left, right, current_gini_index):
    left_probablity = float(len(left)) / (len(left) + len(right))
    right_probabilty = 1 - left_probablity
    information_gain = current_gini_index - left_probablity * gini_index(left) - right_probabilty * gini_index(right)
    return information_gain


"""
        This function will find the best splitting criteria based on current set
        """
def splitting_data_criteria(rows):
    best_gain = 0
    best_question = None
    current_gini_index = gini_index(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = If_question(col, val)
            true_rows, false_rows = division_of_set(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = information_gain(true_rows, false_rows, current_gini_index)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


"""
     Building the decision tree
    """
def decision_tree(rows):
    information_gain, question = splitting_data_criteria(rows)

    if information_gain == 0:
        return final_node(rows)
    right_rows, left_rows = division_of_set(rows, question)
    right_tree = decision_tree(right_rows)
    left_tree = decision_tree(left_rows)
    return current_node(question, right_tree, left_tree)


"""
    Printing the whole decision tree
    """
def output_tree(root, space=""):
    if isinstance(root, final_node):
        li = list(root.final_value.keys())
        for keys in li:
            if keys == "Muffin":
                print(space, "class = [ 1 ] ")
            else:
                print(space,"class = [ 0 ]")
        return
    print(space + str(root.question))
    output_tree(root.right_tree, space + "  ")
    print(space + 'else:')
    output_tree(root.left_tree, space + "  ")


"""
     This functions prints the final node of the decision tree
    """
def final_node_print(type_values):
    class_label = []
    for label in type_values.keys():
        class_label.append(label)
    return class_label


"""
    This function will classify the validation data to find correct class label
    """
def traverse(row, node):
    if isinstance(node, final_node):
        return node.final_value

    if node.question.best(row):
        return traverse(row, node.right_tree)
    else:
        return traverse(row, node.left_tree)


def main():

    # Reading and converting the training csv file to desired state
    food_training_data = pd.read_csv("Recipes_For_Release_2181_v202.csv")
    food_training_data = food_training_data[["FlourOrOats", "Milk", "Sugar", "Butter or Margarine", "Egg",
                                             "Baking Powder", "Vanilla", "Salt", "Baking Soda", "Cream of tartar",
                                             "cinnamon",
                                             "allspice", "nutmeg", "ginger", "Canned Pumpkin_or_Fruit",
                                             "Apple Pie Spice",
                                             "ChocChips", "ChoppedWalnuts", "BrownSugarOrHoney",
                                             "Chopped Pears or Fruit", "Vegetable Oil",
                                             "Water", "Powdered Sugar", "Yogurt", "NUTS", "Type"]]

    training_data = food_training_data.values.tolist()

    # Starting creating decision tree
    decision_tree_object = decision_tree(training_data)

    # Printing decision tree
    output_tree(decision_tree_object)

    # HW_05_Lokare_Hanmant_Classifer.my_classifier(decision_tree_object)

    # Reading and converting the validation csv file to desired state
    food_validation_data = pd.read_csv("Recipes_For_VALIDATION_2181_RELEASED_v202.csv")
    food_validation_data = food_validation_data[["FlourOrOats", "Milk", "Sugar", "Butter or Margarine", "Egg",
                                                 "Baking Powder", "Vanilla", "Salt", "Baking Soda", "Cream of tartar",
                                                 "cinnamon",
                                                 "allspice", "nutmeg", "ginger", "Canned Pumpkin_or_Fruit",
                                                 "Apple Pie Spice",
                                                 "ChocChips", "ChoppedWalnuts", "BrownSugarOrHoney",
                                                 "Chopped Pears or Fruit", "Vegetable Oil",
                                                 "Water", "Powdered Sugar", "Yogurt", "NUTS", "Type"]]

    validation_data = food_validation_data.values.tolist()
    validation_data_type_list = []
    for row in validation_data:
        validation_data_type_list.append(list(final_node_print(traverse(row, decision_tree_object))))

    # appending results of validation decisions to Myclassification.csv file
    file_pointer = open("HW_05_Lokare_Hanmant_MyClassifications.csv", "w+")
    file_pointer.write("Type\n")
    for i in range(len(validation_data_type_list[0])):
        if validation_data_type_list[0][i] == "Muffin":
            file_pointer.write("1")
        else:
            file_pointer.write("0")
        file_pointer.write('\n')



if __name__ == '__main__':
    main()
