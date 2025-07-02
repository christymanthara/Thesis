import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd

def graph_ranks(avranks, names, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.
        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = 0
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # get pairs of non significant methods

        def get_lines(sums, hsd):
            # get all pairs
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            # remove not significant
            notSig = [(i, j) for i, j in allpairs
                      if abs(sums[i] - sums[j]) <= hsd]
            # keep only longest

            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]


    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    if cd and cdmethod is None:
        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2),
              (begin, distanceh - bigtick / 2)],
             linewidth=0.7)
        line([(end, distanceh + bigtick / 2),
              (end, distanceh - bigtick / 2)],
             linewidth=0.7)
        text((begin + end) / 2, distanceh - 0.05, "CD",
             ha="center", va="bottom")

        # no-significance lines
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start),
                      (rankpos(ssums[r]) + side, start)],
                     linewidth=2.5)
                start += height

        draw_lines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod] - cd)
        end = rankpos(avranks[cdmethod] + cd)
        line([(begin, cline), (end, cline)],
             linewidth=2.5)
        line([(begin, cline + bigtick / 2),
              (begin, cline - bigtick / 2)],
             linewidth=2.5)
        line([(end, cline + bigtick / 2),
              (end, cline - bigtick / 2)],
             linewidth=2.5)

    if filename:
        print_figure(fig, filename, **kwargs)

    return fig

def analyze_embedding_methods(csv_file_path, comparison_col_name=None, save_plot=None):
    """
    Analyze embedding methods from CSV file and create Critical Difference diagram.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    comparison_col_name (str, optional): Name of the column containing comparison names.
                                       If None, assumes first column is comparison column.
    save_plot (str, optional): If provided, saves the plot to this filename
    
    Returns:
    dict: Results including best method, ranks, and statistical significance
    """
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV file: {csv_file_path}")
        print(f"Data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Identify the comparison column (first column by default)
    if comparison_col_name is None:
        comparison_col_name = df.columns[0]
    
    # Extract method columns (all columns except the comparison column)
    methods = [col for col in df.columns if col != comparison_col_name]
    
    print(f"\nComparison column: '{comparison_col_name}'")
    print(f"Methods found: {methods}")
    print(f"Number of datasets: {len(df)}")
    print(f"Number of methods: {len(methods)}")
    
    # Extract performance scores
    try:
        scores = df[methods].values
        # Check if all values are numeric
        if not np.all(np.isfinite(scores)):
            print("Warning: Some values are not numeric or contain NaN/Inf")
            print("Please check your data.")
            return None
    except Exception as e:
        print(f"Error extracting scores: {e}")
        return None

    
    print("\nPerformance Scores:")
    print(df)
    print("\n" + "="*60)

    # Calculate ranks for each dataset comparison (lower rank = better performance)
    # Since higher scores are better, we need to reverse the ranking
    ranks = np.zeros_like(scores)
    for i, row in enumerate(scores):
        # Rank in descending order (best performance gets rank 1)
        ranks[i] = rankdata(-row)

    print("\nRanks per dataset (1=best, {}=worst):".format(len(methods)))
    rank_df = pd.DataFrame(ranks, columns=methods, index=df[comparison_col_name])
    print(rank_df)
    print("\n" + "="*60)

    # Calculate average ranks
    avg_ranks = np.mean(ranks, axis=0)
    print("\nAverage Ranks:")
    for i, method in enumerate(methods):
        print(f"{method}: {avg_ranks[i]:.3f}")

    print("\n" + "="*60)

    # Calculate critical difference
    n_datasets = len(df)
    cd = compute_CD(avg_ranks, n_datasets, alpha="0.05", test="nemenyi")
    print(f"\nCritical Difference (α=0.05): {cd:.3f}")

    # Sort methods by average rank for better visualization
    sorted_indices = np.argsort(avg_ranks)
    sorted_methods = [methods[i] for i in sorted_indices]
    sorted_avg_ranks = [avg_ranks[i] for i in sorted_indices]

    print(f"\nMethods ranked by performance (best to worst):")
    for i, (method, rank) in enumerate(zip(sorted_methods, sorted_avg_ranks)):
        print(f"{i+1}. {method}: {rank:.3f}")

    print("\n" + "="*60)
    print("Statistical Significance Analysis:")
    print(f"Two methods are significantly different if their average rank difference > {cd:.3f}")

    # Check which methods are significantly different from the best
    best_rank = sorted_avg_ranks[0]
    significantly_worse = []
    print(f"\nMethods significantly worse than {sorted_methods[0]} (rank {best_rank:.3f}):")
    for method, rank in zip(sorted_methods[1:], sorted_avg_ranks[1:]):
        if rank - best_rank > cd:
            print(f"- {method} (rank {rank:.3f}, difference: {rank - best_rank:.3f})")
            significantly_worse.append(method)
    
    if not significantly_worse:
        print("No methods are significantly worse than the best method.")

    print("\n" + "="*60)
    print("Creating Critical Difference Diagram...")

    # Create the CD diagram
    fig = graph_ranks(avg_ranks, methods, cd=cd, width=10, textspace=2)
    plt.suptitle('Critical Difference Diagram for Embedding Methods\n(Lower rank = better performance)', 
              fontsize=14, y=0.95)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_plot}")
    
    plt.show()

    print("\nConclusion:")
    print(f"The best embedding method is: {sorted_methods[0]} (average rank: {sorted_avg_ranks[0]:.3f})")
    print(f"Methods connected by horizontal lines are not significantly different.")
    
    # Return results as dictionary
    results = {
        'best_method': sorted_methods[0],
        'best_rank': sorted_avg_ranks[0],
        'all_methods_ranked': list(zip(sorted_methods, sorted_avg_ranks)),
        'critical_difference': cd,
        'significantly_worse_methods': significantly_worse,
        'raw_data': df,
        'ranks': rank_df,
        'average_ranks': dict(zip(methods, avg_ranks))
    }
    
    return results


# Example usage function
def run_analysis_example():
    """
    Example of how to use the analysis function with your data
    """
    # Create sample data file (you can skip this if you have your own CSV)
    sample_data = {
        'Source Comparison': [
            'macosko_2015 vs shekhar_2016',
            'hrvatin_2018 vs chen_2017', 
            'baron_2016h vs segerstolpe_transformed',
            'segerstolpe_transformed vs muraro_transformed',
            'muraro_transformed vs xin_2016',
            'xin_2016 vs wang_transformed'
        ],
        'Pavlin t-SNE': [0.977, 0.999, 0.980, 0.960, 0.968, 0.977],
        'pca': [0.969, 0.999, 0.983, 0.992, 0.965, 0.993],
        'pca_harmony': [0.968, 0.999, 0.983, 0.992, 0.966, 0.989],
        'scANVI': [0.987, 1.000, 0.988, 0.968, 0.970, 0.987],
        'scVI': [0.973, 0.999, 0.981, 0.951, 0.965, 0.979],
        'uce': [0.952, 0.997, 0.951, 0.905, 0.918, 0.947],
        'umap': [0.863, 0.926, 0.796, 0.772, 0.893, 0.876]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('embedding_comparison.csv', index=False)
    print("Sample CSV file created: 'embedding_comparison.csv'")
    
    # Run the analysis
    # results = analyze_embedding_methods('embedding_comparison.csv', 
    #                                   save_plot='embedding_cd_diagram.png')
    results = analyze_embedding_methods("/shared/home/christy.jo.manthara/oneliner/knn_reference_cv_results_comprehensive.csv", 
                                      save_plot='embedding_cd_diagram.png')
    return results


# Usage examples:
print("="*80)
print("EMBEDDING METHOD COMPARISON TOOL")
print("="*80)
print("\nTo use this tool with your CSV file:")
print("1. results = analyze_embedding_methods('your_file.csv')")
print("2. results = analyze_embedding_methods('your_file.csv', 'YourComparisonColumnName')")
print("3. results = analyze_embedding_methods('your_file.csv', save_plot='my_plot.png')")
print("\nRunning example with sample data...")
print("="*80)

# Run example
if __name__ == "__main__":
    example_results = run_analysis_example()