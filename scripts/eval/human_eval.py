import pandas as pd
import argparse
import os
import numpy as np
import json
from scipy import stats
from statsmodels.stats.inter_rater import fleiss_kappa


def main(args):
    result = {}
    # Identify score columns (any columns besides the original ones and the experiment column)
    score_cols = ["realism", "relevance"]

    # read the labelled data
    labelled_data = pd.read_csv(args.labelled_data_filepath)
    
    # read all the annotate data
    annotate_data_list = []
    annotate_data_agg = pd.DataFrame()
    for file in os.listdir(args.annotate_data_folder):
        if file.endswith(".csv"):
            annotate_data = pd.read_csv(os.path.join(args.annotate_data_folder, file))
            annotate_data['exp'] = labelled_data['exp']
            annotate_data_agg = pd.concat([annotate_data_agg, annotate_data])
            annotate_data_list.append(annotate_data)
    
    # calculate the average scores and standard deviation of the annotate data group by the exp column in labelled data
    avg_scores = annotate_data_agg.groupby('exp')[score_cols].mean()
    std_scores = annotate_data_agg.groupby('exp')[score_cols].std()

    # convert avg_scores and std_scores to a dictionary
    avg_scores_dict = avg_scores.to_dict(orient='index')
    std_scores_dict = std_scores.to_dict(orient='index')
    result['avg_scores'] = avg_scores_dict
    result['std_scores'] = std_scores_dict

    # compute statistical tests per score_col: ours vs other experiments (Welch's t-test)
    t_stat_result = {}
    for exp in labelled_data['exp'].unique():
        if exp == 'ours':
            continue
        t_stat_result[exp] = {}
        for col in score_cols:
            ours_vals = annotate_data_agg.loc[annotate_data_agg['exp'] == 'ours', col].dropna()
            other_vals = annotate_data_agg.loc[annotate_data_agg['exp'] == exp, col].dropna()
            if len(ours_vals) == 0 or len(other_vals) == 0:
                t_stat_result[exp][col] = {"t_stat": None, "p_value": None}
                print(f"{col} | ours vs {exp}: insufficient data for t-test")
                continue
            t_stat, p_value = stats.ttest_ind(ours_vals, other_vals, equal_var=False)
            t_stat_result[exp][col] = {"t_stat": float(t_stat), "p_value": float(p_value)}
            print(f"{col} | ours vs {exp}: t-statistic={t_stat:.4f}, p-value={p_value:.4g}")
            if p_value < 0.05:
                print(f"{col} | {exp} is significantly different from ours")
            else:
                print(f"{col} | {exp} is not significantly different from ours")
    result['t_stat_result'] = t_stat_result
    
    # calculate fleiss kappa of the agreement between each annotator
    fleiss_kappa_result = {}
    if len(annotate_data_list) >= 2:
        for score_col in score_cols:
            merged = None
            rater_cols = []
            for i, df in enumerate(annotate_data_list, start=1):
                if 'idx' in df.columns:
                    key_col = 'idx'
                    df_keyed = df[[key_col, score_col]].copy()
                else:
                    df_keyed = df.reset_index().rename(columns={'index': 'idx'})[['idx', score_col]].copy()
                    key_col = 'idx'
                rater_name = f"score_{i}"
                df_keyed = df_keyed.rename(columns={score_col: rater_name})
                if merged is None:
                    merged = df_keyed
                else:
                    merged = merged.merge(df_keyed, on=key_col, how='inner')
                rater_cols.append(rater_name)
            if merged is not None and len(rater_cols) > 0:
                for rc in rater_cols:
                    merged[rc] = merged[rc].astype(str)
                categories = sorted(set().union(*[set(merged[rc].dropna().unique()) for rc in rater_cols]))
                rating_matrix = np.array([
                    [((row[rater_cols].dropna()) == cat).sum() for cat in categories]
                    for _, row in merged.iterrows()
                ], dtype=int)
                try:
                    kappa_val = fleiss_kappa(rating_matrix)
                    fleiss_kappa_result[score_col] = kappa_val
                    print(f"Fleiss' kappa for {score_col}: {kappa_val:.4f} "
                          f"(n_items={rating_matrix.shape[0]}, n_raters={len(rater_cols)}, n_categories={len(categories)})")
                except Exception as e:
                    print(f"Failed to compute Fleiss' kappa for {score_col}: {e}")
                    fleiss_kappa_result[score_col] = None
    else:
        print("Not enough annotator files to compute Fleiss' kappa (need at least 2).")
        fleiss_kappa_result = None
    
    result['fleiss_kappa_result'] = fleiss_kappa_result
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate_data_folder", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1218/human_eval/annotate_data", help="path to the annotate data folder")
    parser.add_argument("--labelled_data_filepath", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1218/human_eval/labelled_data.csv", help="path to the labelled data file")
    parser.add_argument("--output_dir", type=str, default="/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1218/human_eval/annotate_data", help="path to the output directory")
    args = parser.parse_args()

    result = main(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "human_eval_result.json"), "w") as f:
        json.dump(result, f, indent=4)
    print(f"human_eval_result saved to {args.output_dir}")


'''
python -m scripts.eval.human_eval \
--annotate_data_folder output_data/gpt5-mini_1218/human_eval/annotate_data \
--labelled_data_filepath output_data/gpt5-mini_1218/human_eval/labelled_data.csv \
--output_dir output_data/gpt5-mini_1218/human_eval/eval_result
'''