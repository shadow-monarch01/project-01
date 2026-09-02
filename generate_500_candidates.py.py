from generate_datasets import generate_candidates

# Generate 500 candidate profiles with custom seed:
generate_candidates(n=500, output_path="data/hiring_500_candidates.csv", seed=999)