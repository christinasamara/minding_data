import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = 'harth'
acceleration_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
def make_dataset():
    all_data = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            all_data = pd.concat([all_data, data])


    #summary_stats = all_data.describe()

def scatter_plots(all_data):
    for label in all_data['label'].unique():
        plt.figure(figsize=(10, 8))
        sns.pairplot(data=all_data[all_data['label'] == label][acceleration_columns], diag_kind='kde')
        plt.suptitle(f'Scatter Plots for Acceleration Columns (Label: {label})', y=1.02)
        plt.show()

def histograms(all_data):
    for label in all_data['label'].unique():
        label_data = all_data[all_data['label'] == label]
        plt.figure(figsize=(12, 8))
        for i, column in enumerate(acceleration_columns):
            plt.subplot(2, 3, i+1)
            plt.hist(label_data[column], bins=20, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of Acceleration ({column}) - Label {label}')
            plt.xlabel('Acceleration')
            plt.ylabel('Frequency')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_label_distribution(all_data):
    activity_counts = all_data['label'].value_counts()
    plt.figure(figsize=(10, 6))
    activity_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Activity Labels')
    plt.xlabel('Activity Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    

def plotting(frame):
    for label in [1,2,3,4,5,6,7,8,13,14,130,140]:
        corr_df = frame[frame['label']==label].drop(['index', 'label', 'Unnamed: 0'], axis=1)
        sns.heatmap(corr_df.corr(),annot=True)
        plt.title(f'heatmap label {label}')
        plt.show()


data = make_dataset()