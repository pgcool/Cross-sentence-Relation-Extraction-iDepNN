from scipy.stats import norm
import tree_rnn

EMB_DIM = 200
HIDDEN_DIM = 200

class RecursiveNNModel(tree_rnn.TreeRNN):
    def train_step_inner(self, x, tree):
        # self._check_input(x, tree)
        # return self._get_final_state(x, tree[:, :-1])
        return self._get_tree_states(x, tree[:, :-1])

    def train_step(self, root_node):
        if all(child is None for child in root_node.children):
            return norm.rvs(size=200)
        x, tree, leaf_computation_order = tree_rnn.gen_nn_inputs(root_node, max_degree=self.degree,
                                   only_leaves_have_vals=False,
                                   with_labels=False)
        tree_states = self.train_step_inner(x, tree)
        return tree_states, leaf_computation_order

def get_model(num_emb, max_degree):
    return RecursiveNNModel(
        num_emb, EMB_DIM, HIDDEN_DIM,
        degree=max_degree)

