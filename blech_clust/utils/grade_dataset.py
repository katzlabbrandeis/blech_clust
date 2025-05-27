import json

def load_grading_metrics():
    with open('grading_metrics.json', 'r') as file:
        return json.load(file)

def calculate_unit_score(unit_count):
    if unit_count < 3:
        return 0.25
    elif 3 <= unit_count <= 6:
        return 0.5
    elif 7 <= unit_count <= 12:
        return 0.75
    else:
        return 1

def calculate_metric_score(sig_fraction):
    return sig_fraction

def calculate_drift_score(sig_fraction):
    return 1 - sig_fraction

def calculate_elbo_score(best_change):
    if best_change <= 1:
        return 1
    elif 2 <= best_change <= 4:
        return 0.66
    elif 5 <= best_change <= 8:
        return 0.33
    else:
        return 0

# Example usage
metrics = load_grading_metrics()
print(calculate_unit_score(5))
print(calculate_metric_score(0.8))
print(calculate_drift_score(0.2))
print(calculate_elbo_score(3))
