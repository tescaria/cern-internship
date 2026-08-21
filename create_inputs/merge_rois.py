import os
import csv

input_dir = "/eos/user/e/exochell/traccc/traccc_athena_plots/g200/traccc-athena/data/roiInputMuon"
output_dir = "/eos/user/t/tcostaes/traccc_outputs/roiInputMuon_merged2"

os.makedirs(output_dir, exist_ok=True)

start_event = 5
end_event = 35

skip_events = {13, 29}

new_event = 0

for event in range(start_event, end_event + 1, 2):

    if event in skip_events:
        print(f"Skipping {event} + {event+1}")
        continue

    event1 = f"event{event:09d}-cells.csv"
    event2 = f"event{event+1:09d}-cells.csv"

    file1 = os.path.join(input_dir, event1)
    file2 = os.path.join(input_dir, event2)

    # Stop if there isn't a pair
    if not os.path.exists(file2):
        break

    output_file = os.path.join(
        output_dir, f"event{new_event:09d}-cells.csv"
    )

    with open(output_file, "w", newline="") as out:
        writer = None

        for filename in [file1, file2]:
            with open(filename, "r", newline="") as f:
                reader = csv.reader(f)

                header = next(reader)

                if writer is None:
                    writer = csv.writer(out)
                    writer.writerow(header)

                for row in reader:
                    writer.writerow(row)

    print(f"{event1} + {event2} -> {output_file}")
    new_event += 1