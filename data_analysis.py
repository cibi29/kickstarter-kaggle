import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.0f}'.format


def first_pass_feature_filter(data):
    drop_list = ['suspended', 'live', 'undefined']
    data = data[~data.isin(drop_list)]
    y = data.state
    drop_list = ['name', 'state', 'ID']
    X = data.drop(drop_list, axis=1)
    return X, y


def count_plot(y):
    sns.countplot(y, label="Count")
    plt.show()

def main():
    csv_filepath = os.path.join('kickstarter-projects', 'ks-projects-201801.csv')
    data = pd.read_csv(csv_filepath)

    # Dividing the data into two sets (domain and label) and removing
    # the obvious ones that we don't need.
    X, y = first_pass_feature_filter(data)
    # remove_irrelevant_data(X, y)
    count_plot(y)


if __name__ == '__main__':
    main()