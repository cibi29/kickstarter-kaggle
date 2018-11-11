import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)


def first_pass_feature_filter(data):
    Y = data.state
    drop_list = ['name', 'state', 'ID']
    X = data.drop(drop_list, axis=1)
    return X, Y


def count_plot(Y):
    sns.countplot(Y, label="Count")
    plt.show()


def main():
    csv_filepath = os.path.join('kickstarter-projects', 'ks-projects-201801.csv')
    data = pd.read_csv(csv_filepath)
    # Dividing the data into two sets (domain and label) and removing
    # the obvious ones that we don't need.
    X, Y = first_pass_feature_filter(data)

    count_plot(Y)


if __name__ == '__main__':
    main()