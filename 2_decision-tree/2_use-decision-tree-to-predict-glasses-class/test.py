from tree import create_decision_tree


def create_dataset(filename):
    with open(filename) as fp:
        dataset = [line.strip().split('\t') for line in fp.readlines()]
    feature_list = ['age', 'prescript', 'astigmatic', 'tearRate']

    return dataset, feature_list


if __name__ == '__main__':
    dataset, feature_list = create_dataset('./data/lenses.txt')
    decision_tree = create_decision_tree(dataset, feature_list)
    print(decision_tree)
