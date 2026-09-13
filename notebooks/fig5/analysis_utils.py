import pandas as pd
import numpy as np
import os, pickle

import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform, cosine, cdist
from scipy.stats import spearmanr

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import warnings


def get_comorb_drugs_data_from_primekg(d1_drugs, d1_name, d2_list, d2_list_names, primekg_indication, drug_metadata, normalized_rank_drugbank, ddi_classes, to_delete_classes, mondo_names_supplement, drug_ind_to_name):
    """
    For each comorbidity (d2_list) of the disease of interest (name: d1_name), 
    take all indications of it as provided by PrimeKG (searching for MONDO ID), and iterate through all drug combinations,
    with drugs of the disease of interest provided also by PrimeKG (searching for name).
    Get the DDI scores and extract one "best" drug combination for each comorbid pair, 
    defined as the one with the highest (among all drug combs for the comorbid pair) lowest (among all DDI classes) normalized rank.
    """
    comorb_drugs_data_primekg_disease = defaultdict(list)
    
    for d2, d2_name in zip(d2_list, d2_list_names):
        d2 = str(d2)
        d2_drugs = primekg_indication[
            (
                (primekg_indication["x_id"] == d2) | \
                (primekg_indication["x_id"].str.contains(f"_{d2}_")) | \
                (primekg_indication["x_id"].str.contains(f"_{d2}$")) | \
                (primekg_indication["x_id"].str.contains(f"^{d2}_")) | \
                (primekg_indication["x_id"].str.contains(f"^{d2}$"))
            ) & \
            (primekg_indication["y_type"] == "drug")
        ]["y_id"].unique()

        # assert that all drugs are in metadata
        if set(d2_drugs) - set(drug_metadata["node_id"].dropna().values) != set():
            print(set(d2_drugs) - set(drug_metadata["node_id"].dropna().values))
        d2_drugs = list(set(d2_drugs) & set(drug_metadata["node_id"].dropna().values))

        # if at least 1 drug for both of the two comorbid diseases is in metadata, look up ddi scores

        # Only keep the best drug combination for each comorbid pair
        if len(d2_drugs) > 0:
            no_valid_sample = True  # use this flag to avoid mistakenly appending the previous drug pair when drug2 = drug1 happens to be the only drug 2
            lowest_highest_ddi_score = 1
            for drug1 in d1_drugs:
                drug_1_ind = drug_metadata[drug_metadata["node_id"] == drug1].index.values[0]
                for drug2 in d2_drugs:
                    if drug1 == drug2:
                        continue
                    no_valid_sample = False
                    drug_2_ind = drug_metadata[drug_metadata["node_id"] == drug2].index.values[0]
                    ddi = normalized_rank_drugbank[:, drug_1_ind, drug_2_ind]
                    if ddi.max() < lowest_highest_ddi_score:
                        best_drug_1_ind = drug_1_ind
                        best_drug_2_ind = drug_2_ind
                        best_ddi = ddi
                        lowest_highest_ddi_score = ddi.max()
            
            if not no_valid_sample:
                if d2_name != d2_name:  # i.e. nan
                    d2_name = mondo_names_supplement[d2]
                drug1_name = drug_ind_to_name[best_drug_1_ind]
                drug2_name = drug_ind_to_name[best_drug_2_ind]
                comorb_drugs_data_primekg_disease["(Disease 1; Disease 2; Drug 1; Drug 2)"].append(f"{d1_name}; " + d2_name.lower() + "; " + str(drug1_name.lower()) + "; " + str(drug2_name.lower()))
                comorb_drugs_data_primekg_disease["DDI"].append(best_ddi)
            
    comorb_drugs_data_primekg_disease_df = pd.DataFrame(comorb_drugs_data_primekg_disease)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comorb_drugs_data_primekg_disease_df[ddi_classes] = comorb_drugs_data_primekg_disease_df["DDI"].apply(pd.Series)
    comorb_drugs_data_primekg_disease_df_filtered = comorb_drugs_data_primekg_disease_df.drop(columns = to_delete_classes + ["DDI"]).set_index("(Disease 1; Disease 2; Drug 1; Drug 2)")
    
    # if we also want to put all quadruples with the same drug pair together
    comorb_drugs_data_primekg_disease_df_filtered["temp_drug_pair"] = pd.Series(comorb_drugs_data_primekg_disease_df_filtered.index).apply(lambda x: "; ".join(x.split("; ")[-2:])).values
    comorb_drugs_data_primekg_disease_df_filtered = comorb_drugs_data_primekg_disease_df_filtered.reset_index().groupby("temp_drug_pair").aggregate(list)
    comorb_drugs_data_primekg_disease_df_filtered["(Disease 1; Disease 2; Drug 1; Drug 2)"] = comorb_drugs_data_primekg_disease_df_filtered["(Disease 1; Disease 2; Drug 1; Drug 2)"].apply(
        lambda lst: "\n".join(lst)
    )
    comorb_drugs_data_primekg_disease_df_filtered = comorb_drugs_data_primekg_disease_df_filtered.set_index("(Disease 1; Disease 2; Drug 1; Drug 2)").applymap(lambda lst: lst[0])
    
    comorb_drugs_data_primekg_disease_df_filtered.to_csv(f"{d1_name.replace(', ', '_').replace(' ', '_')}_comorb_drugs_ddi_scores.csv", index=False)
        
    return comorb_drugs_data_primekg_disease, comorb_drugs_data_primekg_disease_df_filtered


def get_comorb_drugs_data_for_one_comorb_known_drugs(d1_drugs, d1_name, d2_drug_names, d2_name, drug_metadata, normalized_rank_drugbank, ddi_classes, to_delete_classes, drug_ind_to_name):
    """
    Get the DDI scores for all drug combinations between drugs indicated for disease of interest and a set of known drugs indicated for a comorbid disease.
    """
    comorb_drugs_data = defaultdict(list)
    
    # assert that all drugs are in metadata
    if set(d2_drug_names) - set(drug_metadata["node_name"].dropna().str.lower().values) != set():
        print(set(d2_drug_names) - set(drug_metadata["node_name"].dropna().str.lower().values))
    d2_drug_names = list(set(d2_drug_names) & set(drug_metadata["node_name"].dropna().str.lower().values))

    # look up ddi scores
    for drug1 in d1_drugs:
        if drug1.startswith("DB"):
            drug_1_ind = drug_metadata[drug_metadata["node_id"] == drug1].index.values[0]
        else:
            drug_1_ind = int(drug1)
        for drug2_name in d2_drug_names:
            drug_2_ind = drug_metadata[drug_metadata['node_name'].str.lower() == drug2_name.lower()].index.values[0]
            if drug_1_ind == drug_2_ind:
                print("overlap")
                continue
            ddi = normalized_rank_drugbank[:, drug_1_ind, drug_2_ind]
            drug1_name = drug_ind_to_name[drug_1_ind]
            comorb_drugs_data["(Disease 1; Disease 2; Drug 1; Drug 2)"].append(f"{d1_name}; " + d2_name.lower() + "; " + str(drug1_name.lower()) + "; " + str(drug2_name.lower()))
            comorb_drugs_data["DDI"].append(ddi)

    comorb_drugs_data_df = pd.DataFrame(comorb_drugs_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comorb_drugs_data_df[ddi_classes] = comorb_drugs_data_df["DDI"].apply(pd.Series)
    comorb_drugs_data_df_filtered = comorb_drugs_data_df.drop(columns = to_delete_classes + ["DDI"]).set_index("(Disease 1; Disease 2; Drug 1; Drug 2)")
    comorb_drugs_data_df_filtered.to_csv(f"{d1_name.replace(', ', '_').replace(' ', '_')}_VS_{d2_name.replace(', ', '_').replace(' ', '_')}_drugs_ddi_scores.csv")
        
    return comorb_drugs_data, comorb_drugs_data_df_filtered


def remove_indices_with_adverse_ddi_labels(scores_sorted_indices, ddi_sample_names, drug_name_to_ind, db_polypharmacy, valid_ddi_class_indices):
    """
    Remove those indices that have adverse DDI labels. Needs global var `db_polypharmacy`.
    """
    is_to_remove = []
    for i, idx in enumerate(scores_sorted_indices):
        disease_drug_list = ddi_sample_names[idx].split('; ')
        drug_1_name = disease_drug_list[-2]
        drug_2_name = disease_drug_list[-1]
        drug_1_ind = drug_name_to_ind[drug_1_name]
        drug_2_ind = drug_name_to_ind[drug_2_name]
        ddi_adverse_labels = db_polypharmacy[(db_polypharmacy['drug_index_1'] == drug_1_ind) & (db_polypharmacy['drug_index_2'] == drug_2_ind)]['label_indexed'].values
        if bool(set(ddi_adverse_labels) & set(valid_ddi_class_indices)):
            print(disease_drug_list, drug_1_ind, drug_2_ind, ddi_adverse_labels)
            is_to_remove.append(i)
    return np.delete(scores_sorted_indices, is_to_remove)


def get_highest_and_lowest_drug_pairs(ddi_profile_df, drug_name_to_ind, db_polypharmacy, valid_ddi_class_indices, highest_criterion="median", lowest_criterion="highest", num_select=10, return_sorted_dfs=False):
    """
    Args:
        ddi_profile_df: 
            Indices are sample names (e.g. "Disease 1; Disease 2; Drug 1; Drug 2"), column names are DDI class names. 
            The index must have a name (so that `ddi_profile_df.index.name` is valid).
        highest/lowest_criterion:
            The criterion used to select the highest/lowest 10 drug pairs. Select from "median", "highest", "highest_5_mean".
    """
    # lowest and highest comorb disease pairs + drug pairs
    # NOTE: HIGH is bad, LOW is good
    median_row_scores = ddi_profile_df.apply(np.median, axis=1).values
    highest_row_scores = ddi_profile_df.apply(max, axis=1).values
    highest_5_mean_row_scores = ddi_profile_df.apply(lambda x: np.mean(np.partition(x, len(x)-5)[-5:]), axis=1).values
    
    # NOTE: The reason for us to trim down the number of drug pairs to 50 here is that we want to make the next step (`remove_indices_with_adverse_ddi_labels`) more efficient
    median_row_scores_highest_indices = np.argsort(median_row_scores)[-50:]
    highest_row_scores_highest_indices = np.argsort(highest_row_scores)[-50:]
    highest_5_mean_row_scores_highest_indices = np.argsort(highest_5_mean_row_scores)[-50:]
    median_row_scores_lowest_indices = np.argsort(median_row_scores)[:50]
    highest_row_scores_lowest_indices = np.argsort(highest_row_scores)[:50]
    highest_5_mean_row_scores_lowest_indices = np.argsort(highest_5_mean_row_scores)[:50]
    
    # filter out those predicted lowest (safest) drug pairs where there are actually DDIs in the ground truth
    median_row_scores_lowest_indices_filtered = remove_indices_with_adverse_ddi_labels(median_row_scores_lowest_indices, ddi_profile_df.index.values, drug_name_to_ind, db_polypharmacy, valid_ddi_class_indices)
    highest_row_scores_lowest_indices_filtered = remove_indices_with_adverse_ddi_labels(highest_row_scores_lowest_indices, ddi_profile_df.index.values, drug_name_to_ind, db_polypharmacy, valid_ddi_class_indices)
    highest_5_mean_row_scores_lowest_indices_filtered = remove_indices_with_adverse_ddi_labels(highest_5_mean_row_scores_lowest_indices, ddi_profile_df.index.values, drug_name_to_ind, db_polypharmacy, valid_ddi_class_indices)

    # take highest and lowest 10 filtered from median (by default) and highest (by default), respectively
    if highest_criterion == "median":
        highest = ddi_profile_df.iloc[median_row_scores_highest_indices[-num_select:]]
    elif highest_criterion == "highest":
        highest = ddi_profile_df.iloc[highest_row_scores_highest_indices[-num_select:]]
    elif highest_criterion == "highest_5_mean":
        highest = ddi_profile_df.iloc[highest_5_mean_row_scores_highest_indices[-num_select:]]
    else:
        raise NotImplementedError
    
    if lowest_criterion == "median":
        lowest = ddi_profile_df.iloc[median_row_scores_lowest_indices_filtered[:num_select]]
    elif lowest_criterion == "highest":
        lowest = ddi_profile_df.iloc[highest_row_scores_lowest_indices_filtered[:num_select]]
    elif lowest_criterion == "highest_5_mean":
        lowest = ddi_profile_df.iloc[highest_5_mean_row_scores_lowest_indices_filtered[:num_select]]
    else:
        raise NotImplementedError
    
    drug_combos_d1_highest = highest.index.values
    drug_combos_d1_lowest = lowest.index.values

    highest_long = highest.reset_index()
    highest_long = pd.melt(highest_long, id_vars=ddi_profile_df.index.name, value_vars=ddi_profile_df.columns, var_name="ddi_class")
    
    lowest_long = lowest.reset_index()
    lowest_long = pd.melt(lowest_long, id_vars=ddi_profile_df.index.name, value_vars=ddi_profile_df.columns, var_name="ddi_class")
    
    outputs = (highest, lowest, highest_long, lowest_long)
    
    if return_sorted_dfs:
        median_row_scores_indices_sorted = np.argsort(median_row_scores)
        highest_row_scores_indices_sorted = np.argsort(highest_row_scores)
        highest_5_mean_row_scores_indices_sorted = np.argsort(highest_5_mean_row_scores)
        ddi_profile_df_sorted_by_median = ddi_profile_df.iloc[median_row_scores_indices_sorted]
        ddi_profile_df_sorted_by_highest = ddi_profile_df.iloc[highest_row_scores_indices_sorted]
        ddi_profile_df_sorted_by_highest_5_mean = ddi_profile_df.iloc[highest_5_mean_row_scores_indices_sorted]
        outputs += (ddi_profile_df_sorted_by_median, ddi_profile_df_sorted_by_highest, ddi_profile_df_sorted_by_highest_5_mean)
    
    return outputs


def get_highest_percent_drugs(comorb_drugs_df, organ_class_mapping, organs_of_interest, threshold=0.9):
    """
    Get the DDI profile for DDIs with normalized ranks greater than {threshold} for each sample ((disease 1, disease 2, drug 1, drug2)) in {comorb_drugs_df}. 
    """
    # get percent ddi classes that are > 0.9 for each of the top 10 drugs
    classes_of_interest = []

    drugs_above_threshold = defaultdict(dict)
    drugs_above_threshold_num = defaultdict(lambda : defaultdict(int))
    rows_included = []

    for _, row in comorb_drugs_df.iterrows():
        for ddi_class_name, value in row.items():
            if value > threshold:
                organs = organ_class_mapping[ddi_class_name]
                classes_of_interest.append((str(row.name), ddi_class_name, organs, value))
                drugs_above_threshold[str(row.name)][(ddi_class_name, organs)] = value
                has_been_others = False
                for organ in organs.split(", "):
                    if organ not in organs_of_interest:
                        organ = "others/general"
                    if organ != "others/general" or not has_been_others:
                        drugs_above_threshold_num[str(row.name)][organ] += 1
                    if organ == "others/general":
                        has_been_others = True
                rows_included.append(str(row.name))
        
        # if the row does not have any risky DDI class, add a placeholder empty dictionary
        if str(row.name) not in rows_included:
            drugs_above_threshold[str(row.name)] = {}
            drugs_above_threshold_num[str(row.name)] = {}
    
    print("Classes of interest " + str(len(classes_of_interest)))
    
    return drugs_above_threshold, drugs_above_threshold_num, classes_of_interest


def add_organs_column(highest_long, lowest_long, organ_class_mapping, organs_of_interest, organ_counts, sample_name_col_name="(Disease 1; Disease 2; Drug 1; Drug 2)", num_least_organ=5):
    """
    Add a column indicating the DDI classes's organs to the DDI profile dataframes in place.
    Args:
        highest_long, lowest_long: Long format of DDI profiles
    """
    organs_list = []

    for _, row in highest_long.iterrows():
        ddi_class_name = str(row["ddi_class"])
        organs = list(set([organ if organ in organs_of_interest else "others/general" for organ in organ_class_mapping[ddi_class_name].split(", ")]))
        organs_list.append(organs)

    highest_long["organ"] = organs_list
    highest_long_exploded = highest_long.explode("organ").reset_index(drop=True)
    for key in organ_counts.keys():
        # check if there are any organs with < num_least_organ points
        if organ_counts[key] < num_least_organ:
            highest_long_exploded["organ"] = highest_long_exploded["organ"].replace(key, "others/general")
    
    if lowest_long is not None:
        assert (highest_long["ddi_class"] != lowest_long["ddi_class"]).sum() == 0
        lowest_long["organ"] = organs_list
        lowest_long_exploded = lowest_long.explode("organ").reset_index(drop=True)
        for key in organ_counts.keys():
            # check if there are any organs with < num_least_organ points
            if organ_counts[key] < num_least_organ:
                lowest_long_exploded["organ"] = lowest_long_exploded["organ"].replace(key, "others/general")
    else:
        lowest_long_exploded = None
            
    return highest_long_exploded, lowest_long_exploded


def make_stripplot(
    ddi_profile_long_df, ddi_profile_below_threshold, ddi_profile_below_threshold_nums, plot_title, custom_palette,
    sample_name_col_name="(Disease 1; Disease 2; Drug 1; Drug 2)", threshold_to_count_ddi=0.9, print_ddi_class_or_organ_num="organ_num",
):
    """
    Args:
        ddi_profile_long_df: Long DDI profile (columns: {sample_name_col_name}, ddi_class, value, organ)
        ddi_profile_below_threshold_nums: Dict[sample_name, Dict[organ, num_ddi_classes_below_threshold]]
        print_ddi_class_or_organ_num: "organ_num" or "ddi_class" (print the number of DDI classes or the DDI classes themselves on the right of the plot)
    """
    sns.set_theme(style="whitegrid")

    # Initialize the figure
    plt.figure(figsize=(10, 15), dpi=300)

    # Show each observation with a scatterplot
    sns.stripplot(
        data=ddi_profile_long_df, x="value", y=sample_name_col_name, hue="organ", 
        palette=custom_palette, dodge=1.0, jitter=False, alpha=.8
    )

    # Put dashed horizontal lines to separate the drug combinations
    y = 0.5
    while y < 10.5:
        plt.hlines(y=y, xmin=-0.5, xmax=3.5, linestyles="dashed", color="grey")
        y += 1

    height = 0.0
    
    if print_ddi_class_or_organ_num == "organ_num":
        for quadruple, organ_counts in ddi_profile_below_threshold_nums.items():
            value_str = ""
            for organ, count in organ_counts.items():
                value_str += organ + ": " + str(count) + "; "
            plt.text(1.05, height, value_str.strip("; "), fontsize=10, va="center")
            height += 1

        plt.text(1.05, height, f"# of DDI classes > {str(threshold_to_count_ddi)} per organ class:")
        
    elif print_ddi_class_or_organ_num == "ddi_class":
        for quadruple, ddi_class_norm_ranks in ddi_profile_below_threshold.items():
            value_str = ""
            for ddi_class_tuple, norm_rank in ddi_class_norm_ranks.items():
                value_str += f"{ddi_class_tuple[0]} ({ddi_class_tuple[1]}, {norm_rank:.4f})\n"
            plt.text(1.05, height, value_str.strip("\n"), fontsize=10, va="center")
            height += 1

        plt.text(1.05, height, f"DDI classes > {str(threshold_to_count_ddi)}:")
        
    else:
         raise NotImplementedError   

    plt.xlim([0, 1])
    plt.ylim([-0.5, 10])
    plt.axvline(x=threshold_to_count_ddi, c="r")
    plt.xlabel("Normalized Rank:\n0 = Lowest predicted associations between drug combination and adverse event\n1 = Highest predicted association between drug combination and adverse event")
    plt.title(plot_title)
    plt.xticks(fontsize=8)
    plt.legend(bbox_to_anchor=(1.8, 1.1), loc='upper right')
    plt.show()
