import json    

file_test = "please-base-scenario_responses_pers_scores"
filename_data = f"results_baseline/{file_test}.txt"
with open(filename_data, "r") as file:
    data = json.load(file)

trait_sums = {}
trait_counts = {}

for trait_list in data:
    for trait_entry in trait_list:
        trait = trait_entry["trait"]
        prediction = trait_entry["prediction"]
        if trait not in trait_sums:
            trait_sums[trait] = 0
            trait_counts[trait] = 0
        trait_sums[trait] += prediction
        trait_counts[trait] += 1

trait_averages = {
    trait: trait_sums[trait] / trait_counts[trait] for trait in trait_sums
}

print("Average predictions for each trait:")
print(trait_averages)
