"""Creates a mixed fullscan-regional directory according to HL-LHC rates"""

from pathlib import Path
import random
import shutil


# input/output directories 
muon_dir = Path("/eos/user/e/exochell/traccc/traccc_athena_plots/g200/traccc-athena/data/roiInputMuon")
jet_dir = Path("/eos/user/t/tcostaes/traccc_outputs/roiInputJets_nodup")
full_dir = Path("/eos/project/a/atlas-eftracking/GPU/ITk_data/traccc_standalone_data/ttbar_mu200")

output_dir = Path("/eos/user/t/tcostaes/traccc_outputs/mixedDir")
total_events = 70
seed = 42

# HL-LHC rates (kHz)
regional_rate = 1000
full_rate = 150
muon_rate = 1.7
jet_rate = 6.5 

# calculate fractions 
# composition of regional part
muon_regional_fraction = muon_rate / (muon_rate + jet_rate)
jet_regional_fraction = jet_rate / (muon_rate + jet_rate)
# regional vs full workload
regional_fraction = regional_rate / (regional_rate + full_rate)
full_fraction = full_rate / (regional_rate + full_rate)
# final fractions 
muon_fraction = regional_fraction * muon_regional_fraction
jet_fraction = regional_fraction * jet_regional_fraction

# calculate event counts
n_muon = round(total_events * muon_fraction)
n_jet = round(total_events * jet_fraction)
n_full = total_events - n_muon - n_jet

print(f"Regional fraction : {regional_fraction:.4%}")
print(f"Full fraction     : {full_fraction:.4%}")
print()

print(f"Muon regional fraction : {muon_regional_fraction:.4%}")
print(f"Jet regional fraction  : {jet_regional_fraction:.4%}")
print()

print("Final fractions:")
print(f"  Muon : {muon_fraction:.4%}")
print(f"  Jet  : {jet_fraction:.4%}")
print(f"  Full : {full_fraction:.4%}")
print()

print(f"Requested events: {total_events}")
print("Events to select:")
print(f"  Muon : {n_muon}")
print(f"  Jet  : {n_jet}")
print(f"  Full : {n_full}")
print(f"  Total: {n_muon + n_jet + n_full}")

def get_events(directory):
    """find all event*-cells.csv files"""
    return sorted(directory.glob("event*-cells.csv"))

def sample_events(directory, n):
    events = get_events(directory)

    if len(events) < n:
        raise RuntimeError(
            f"{directory} contains only {len(events)} events, but {n} are required.")

    return random.sample(events, n)

# create output directory
if output_dir.exists():
    existing_events = list(output_dir.glob("event*-cells.csv"))

    # nice AI add to prevent overwritting the directory :)
    if existing_events:
        raise RuntimeError(
            f"Output directory already contains {len(existing_events)} event files: {output_dir}\n"
            "Please choose another directory or remove the existing files."
        )
else:
    output_dir.mkdir(parents=True)

# select events
random.seed(seed)
muon_events = sample_events(muon_dir, n_muon)
jet_events = sample_events(jet_dir, n_jet)
full_events = sample_events(full_dir, n_full)
mixed_events = muon_events + jet_events + full_events

# randomise order
random.shuffle(mixed_events)

# copy and rename 
for i, source in enumerate(mixed_events):
    destination = output_dir/f"event{i:09d}-cells.csv"
    shutil.copy2(source, destination)

print()
print(f"Successfully created {len(mixed_events)} events in:")
print(f"  {output_dir}")