import pandas as pd
import Nodes
import sys

# input arguments
training_data = pd.read_csv(sys.argv[1])
validation_data = pd.read_csv(sys.argv[2])
testing_data = pd.read_csv(sys.argv[3])
pruning_factor = float(sys.argv[4])

label = 1
col_val = training_data.columns.values
validation_col_val = validation_data.columns.values
testing_col_val = testing_data.columns.values

attribute_val = col_val[:-1]
class_label = col_val[-1]
validation_attribute_val = col_val[:-1]
testing_attribute_val = col_val[:-1]

class_instance = Nodes.getClassLabels(training_data, class_label)
instance = Nodes.getInstances(training_data)
validation_instance = Nodes.getInstances(validation_data)
testing_instance = Nodes.getInstances(testing_data)
root_node = Nodes.build_decsiontree(instance, class_instance, attribute_val, training_data, label)

print('Print DecisionTree : ')
Nodes.print_tree(root_node, 0)

# calculation for pre-pruned accuracy:
prepruned_accr = Nodes.accuracy(root_node, validation_data)


print('\n')
print('Pre-Pruned Accuracy:\n')
print('Number of Training instances = ', len(instance))
print('Number of Training attributes = ', len(attribute_val))
print('Total number of nodes in the DecisionTree = ', Nodes.count_total_nodes(root_node))
print('Number of leaf nodes in the DecisionTree = ', Nodes.count_pnodes(root_node))
print('Accuracy of the model on the training set : ', round(Nodes.accuracy(root_node, training_data) *
                                                                100, 2),'%')
print('\n')
print('Number of Validation instances = ', len(validation_instance))
print('Number of Validation attributes = ', len(validation_attribute_val))
print('Accuracy of the model on the validation set before pruning : ', round(prepruned_accr * 100, 2), '%')
print('\n')
print('Number of Testing instances = ', len(testing_instance))
print('Number of Testing attributes = ', len(testing_attribute_val))
print('Accuracy of the model on the testing set before pruning : ',
      round(Nodes.accuracy(root_node, testing_data) * 100, 2), '%')



prune_nodes_num = int(round(pruning_factor * Nodes.count_npnodes(root_node), 0))
Nodes.getprune_nodes(root_node, prune_nodes_num, prepruned_accr, root_node, validation_data)
prune_nodes = Nodes.nodes_values


print('\n')
print('Post-Pruned Accuracy:\n')
print('Number of Training instances = ', len(instance))
print('Number of Training attributes = ', len(attribute_val))
print('Total number of nodes in the DecisionTree = ', Nodes.count_pruned_nodes(root_node, prune_nodes))
print('Number of leaf nodes in the DecisionTree = ', Nodes.count_after_prune(root_node, prune_nodes))
print('Accuracy of the model on the training set after pruning : ',
      round(Nodes.postPruneAccr(root_node, training_data, prune_nodes) * 100, 2), '%')
print('\n')
print('Number of Validation instances = ', len(validation_instance))
print('Number of Validation attributes = ', len(validation_attribute_val))
print('Accuracy of the model on the validation set after pruning : ',
      round(Nodes.postPruneAccr(root_node, validation_data, prune_nodes) * 100, 2), '%')
print('\n')
print('Number of Testing instances = ', len(testing_instance))
print('Number of Testing attributes= ', len(testing_attribute_val))
print('Accuracy of the model on the testing set after pruning : ',
      round(Nodes.postPruneAccr(root_node, testing_data, prune_nodes) * 100, 2), '%')
input('press Enter')