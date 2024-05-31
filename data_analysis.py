import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as stats

"""
The function anova_dimension_reduction performs the Kruskal-Wallis test on multiple datasets. With the Kruskal-Wallis test
we can statistically prove that different normalization and dimension reduction methods have significant effects on the NDRindex.
"""


def anova_dimension_reduction(datasets):
    alpha = 0.05

    for df, title in datasets:
        df[['Normalization', 'DimensionReduction']] = df['Combination'].str.split('+', expand=True)

        # Kruskal-Wallis Test for Normalization
        grouped_norm = [df['NDRindex'][df['Normalization'] == norm] for norm in df['Normalization'].unique()]
        kruskal_result_norm = stats.kruskal(*grouped_norm)

        # Kruskal-Wallis Test for Dimension Reduction
        grouped_dimred = [df['NDRindex'][df['DimensionReduction'] == dimred] for dimred in
                          df['DimensionReduction'].unique()]
        kruskal_result_dimred = stats.kruskal(*grouped_dimred)

        print("Kruskal-Wallis test results for Normalization methods:")
        print(f"H-statistic: {kruskal_result_norm.statistic}, p-value: {kruskal_result_norm.pvalue}")

        if kruskal_result_norm.pvalue < alpha:
            print(
                f"The normalization method has a statistically significant effect on NDRindex (p-value = {kruskal_result_norm.pvalue:.4f}).")
        else:
            print(
                f"The normalization method does not have a statistically significant effect on NDRindex (p-value = {kruskal_result_norm.pvalue:.4f}).")

        print("\nKruskal-Wallis test results for Dimension Reduction methods:")
        print(f"H-statistic: {kruskal_result_dimred.statistic}, p-value: {kruskal_result_dimred.pvalue}")

        if kruskal_result_dimred.pvalue < alpha:
            print(
                f"The dimension reduction method has a statistically significant effect on NDRindex (p-value = {kruskal_result_dimred.pvalue:.4f}).")
        else:
            print(
                f"The dimension reduction method does not have a statistically significant effect on NDRindex (p-value = {kruskal_result_dimred.pvalue:.4f}).")
        print("---------------")


"""
The function boxplots_dimension_reduction will create boxplots that compare the values of the NDRindex with different
demension reduction methods for multiple datasets.
 """


def boxplots_dimension_reduction(datasets):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    # Iterate over datasets and corresponding axes
    for ax, (df, title) in zip(axes, datasets):
        df[['Normalization', 'DimensionReduction']] = df['Combination'].str.split('+', expand=True)
        sns.boxplot(x='DimensionReduction', y='NDRindex', data=df, ax=ax)
        ax.set_title(f'NDRindex by Dimension Reduction Method\n{title}')
        ax.set_xlabel('Dimension Reduction')
        ax.set_ylabel('NDRindex' if ax == axes[0] else '')  # Only label the y-axis on the first subplot

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    for df, title in datasets:
        df[['Normalization', 'DimensionReduction']] = df['Combination'].str.split('+', expand=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='DimensionReduction', y='NDRindex', data=df)
        plt.title(f'NDRindex by Dimension Reduction Method {title}')
        plt.show()


"""
The function mean_ndr_combination will create barplots of the NDRindex across different normalization methods for multiple datasets.
"""


def mean_ndr_combination(datasets):
    for df, title in datasets:
        df[['Normalization', 'DimensionReduction']] = df['Combination'].str.split('+', expand=True)

        # Calculate mean NDRindex for each combination
        mean_ndrindex = df.groupby(['Normalization', 'DimensionReduction'])['NDRindex'].mean().unstack()

        # Plotting
        mean_ndrindex.plot(kind='bar', figsize=(12, 8))
        plt.title(f'Mean NDRindex for Each Combination of Normalization and Dimension Reduction for {title}')
        plt.ylabel('Mean NDRindex')
        plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    # Iterate over datasets and corresponding axes
    for ax, (df, title) in zip(axes, datasets):
        df[['Normalization', 'DimensionReduction']] = df['Combination'].str.split('+', expand=True)

        # Calculate mean NDRindex for each combination
        mean_ndrindex = df.groupby(['Normalization', 'DimensionReduction'])['NDRindex'].mean().unstack()

        # Plotting on the corresponding axis
        mean_ndrindex.plot(kind='bar', ax=ax)
        ax.set_title(f'Mean NDRindex for Each Combination\nof Normalization and Dimension Reduction\nfor {title}')
        ax.set_ylabel('Mean NDRindex')
        ax.set_xlabel('Normalization')
        ax.legend(title='DimensionReduction')


"""
The function pearson_correlation will calculate whether there is significant correlation between the NDRindex and ARI for multiple datasets.
"""


def pearson_correlation(datasets):
    def interpret_p_value(p_value, alpha=0.1):
        if p_value < alpha:
            return "There is a statistically significant correlation."
        else:
            return "There is no statistically significant correlation."

    for df, title in datasets:
        # Calculate and interpret Pearson correlation for NDRindex vs ARI-hclust
        corr_ari_hclust, p_value_ari_hclust = pearsonr(df['NDRindex'], df['ARI-hclust'])
        interpretation_ari_hclust = interpret_p_value(p_value_ari_hclust)
        print(
            f"Pearson correlation between NDRindex and ARI-hclust: {corr_ari_hclust:.4f} (p-value: {p_value_ari_hclust:.4e}) - {interpretation_ari_hclust}")

        # Calculate and interpret Pearson correlation for NDRindex vs ARI-kmeans
        corr_ari_kmeans, p_value_ari_kmeans = pearsonr(df['NDRindex'], df['ARI-kmeans'])
        interpretation_ari_kmeans = interpret_p_value(p_value_ari_kmeans)
        print(
            f"Pearson correlation between NDRindex and ARI-kmeans: {corr_ari_kmeans:.4f} (p-value: {p_value_ari_kmeans:.4e}) - {interpretation_ari_kmeans}")

        # Calculate and interpret Pearson correlation for NDRindex vs ARI-spectral
        corr_ari_spectral, p_value_ari_spectral = pearsonr(df['NDRindex'], df['ARI-spectral'])
        interpretation_ari_spectral = interpret_p_value(p_value_ari_spectral)
        print(
            f"Pearson correlation between NDRindex and ARI-spectral: {corr_ari_spectral:.4f} (p-value: {p_value_ari_spectral:.4e}) - {interpretation_ari_spectral}")


data1 = pd.read_csv('output_dataframes/data_1.csv')
data2 = pd.read_csv('output_dataframes/data_2.csv')
data3 = pd.read_csv('output_dataframes/data_3.csv')
data = [(data1, "dataset 1"), (data2, "dataset 2"), (data3, "dataset 3")]

boxplots_dimension_reduction(data)
mean_ndr_combination(data)
pearson_correlation(data)
anova_dimension_reduction(data)
