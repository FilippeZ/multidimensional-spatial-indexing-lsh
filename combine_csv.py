import pandas as pd
import os
import numpy as np

# Συνάρτηση για μέτρηση μεγέθους αρχείου σε MB
def measure_file_size_mb(filepath: str) -> float:
    # Επιστροφή μεγέθους αρχείου σε MB
    return os.path.getsize(filepath) / (1024 * 1024)

# Συνάρτηση για δημιουργία DataFrame και μέτρηση μεγέθους
def write_and_measure(df: pd.DataFrame, output_file: str) -> float:
    # Αποθήκευση DataFrame σε CSV αρχείο χωρίς δείκτες
    df.to_csv(output_file, index=False)
    # Μέτρηση μεγέθους του αποθηκευμένου αρχείου
    size_mb = measure_file_size_mb(output_file)
    # Επιστροφή μεγέθους σε MB
    return size_mb

# Συνάρτηση για συνδυασμό CSV αρχείων με στόχο περίπου τα 199 MB
def combine_csv_approx_199mb(csv_files, output_file="data.csv", 
                             target_mb=199.0, tolerance_mb=1.0, max_iter=10):
    # Λίστα για αποθήκευση των DataFrames
    dfs = []
    # Επανάληψη για κάθε CSV αρχείο
    for f in csv_files:
        # Ανάγνωση CSV αρχείου σε DataFrame
        print(f"Reading {f} ...")
        temp_df = pd.read_csv(f, low_memory=False)
        # Προσθήκη DataFrame στη λίστα
        dfs.append(temp_df)
    # Συνδυασμός όλων των DataFrames σε ένα
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Initial combined shape: {combined_df.shape}")
    # Εγγραφή και μέτρηση μεγέθους
    size_mb = write_and_measure(combined_df, output_file)
    print(f"Initial file size: {size_mb:.2f} MB")

    # Έλεγχος αν το μέγεθος είναι ήδη κάτω από τον στόχο
    if size_mb <= target_mb:
        print(f"Final file size {size_mb:.2f} MB is <= target ({target_mb} MB).")
        # Επιστροφή του DataFrame
        return combined_df

    # Αρχικοποίηση μετρητή επαναλήψεων
    iteration = 0
    # Βρόγχος επαναλήψεων μέχρι το μέγιστο όριο
    while iteration < max_iter:
        # Έλεγχος αν το μέγεθος δεν ξεπερνάει το κατώφλι
        if abs(size_mb - target_mb) <= tolerance_mb:
            # Μήνυμα Επιτυχίας
            print(f"File size {size_mb:.2f} MB is within ±{tolerance_mb} MB of {target_mb} MB.")
            # Έξοδος από τον Βρόγχο
            break

        # Έλεγχος αν το μέγεθος έπεσε κάτω από τον στόχο
        if size_mb < target_mb:
            # Μήνυμα Διακοπής
            print(f"File size dropped below target at {size_mb:.2f} MB. Stopping iteration.")
            # Έξοδος από τον Βρόγχο
            break
        
        # Ποσοστό δεδομένων που πρέπει να κρατήσουμε (στόχος ÷ τρέχον μέγεθος)
        fraction = target_mb / size_mb
        # Μείωση κατά 1% για ασφάλεια, ώστε αποφυχθει η υπερβολική συρρίκνωση του αρχείου
        sample_fraction = fraction * 0.99

        # Έλεγχος αν το κλάσμα δεν είναι έγκυρο
        if sample_fraction <= 0.0:
            # Μήνυμα Σφάλματος
            print("sample_fraction is 0 or negative. Not possible to continue.")
            # Έξοδος από τον Βρόγχο
            break

        # Υπολογισμός νέου αριθμού γραμμών       
        new_count = int(len(combined_df) * sample_fraction)
        # Έλεγχος αν ο αριθμός γραμμών είναι πολύ μικρός
        if new_count < 1:
            # Μήνυμα Διακοπής
            print("Only one or zero rows left after sampling. Stopping.")
            # Έξοδος από τον Βρόγχο
            break

        # Εμφάνιση πληροφοριών επανάληψης (τρέχον μέγεθος αρχείου, ποσοστό δειγματοληψίας και πλήθος γραμμών)
        print(f"Iteration {iteration+1}: size = {size_mb:.2f} MB, sampling fraction ~ {sample_fraction:.4f} => {new_count} rows")
        # Επιλέγω τυχαίο υποσύνολο γραμμών από το DataFrame, με βάση τον υπολογισμένο αριθμό (new_count),
        # ώστε να μειωθεί το συνολικό μέγεθος του αρχείου CSV.
        # Η δειγματοληψία είναι τυχαία αλλά αναπαράξιμη (χρησιμοποιώντας σταθερό random_state=42),
        # δηλαδή κάθε φορά που εκτελείται ο ίδιος κώδικας, θα παραχθεί το ίδιο δείγμα,
        # κάτι το οποίο ειναι θεμελιώδες για την συγκεκριμένη εργασία, καθώς διατηρεί την συνέπεια.
        # Το reset_index(drop=True) χρησιμοποιείται για να επαναριθμήσει τους δείκτες των γραμμών,
        # καθώς μετά τη δειγματοληψία οι παλιοί δείκτες δεν είναι συνεχόμενοι και μπορεί να προκαλέσουν προβλήματα σε μεταγενέστερη επεξεργασία.
        # Το drop=True σημαίνει ότι οι παλιοί δείκτες δεν αποθηκεύονται ως νέα στήλη, απλά απορρίπτονται.
        combined_df = combined_df.sample(n=new_count, random_state=42).reset_index(drop=True)
        # Εγγραφή και Μέτρηση Νέου Μεγέθους
        size_mb = write_and_measure(combined_df, output_file)
        # Εκτύπωση Νέου Μεγέθους
        print(f"New file size: {size_mb:.2f} MB")
        # Αύξηση Μετρητή Επαναλήψεων
        iteration += 1

    # Μέτρηση τελικού μεγέθους αρχείου
    final_size = measure_file_size_mb(output_file)
    # Εκτύπωση τελικού μεγέθους
    print(f"Final file size after iteration {iteration}: {final_size:.2f} MB")
    # Εκτύπωση τελικού σχήματος δεδομένων
    print(f"Final shape: {combined_df.shape}")
    # Επιστροφή του τελικού DataFrame
    return combined_df

if __name__ == "__main__":

    # Λίστα με τα CSV αρχεία 
    csv_files = [
        "cms_hospital_patient_satisfaction_2016.csv",
        "cms_hospital_patient_satisfaction_2017.csv",
        "cms_hospital_patient_satisfaction_2018.csv",
        "cms_hospital_patient_satisfaction_2019.csv",
        "cms_hospital_patient_satisfaction_2020.csv",
    ]
    # Όνομα αρχείου εξόδου
    output_file = "data.csv"

    # Κλήση της κύριας συνάρτησης
    final_df = combine_csv_approx_199mb(
        csv_files,
        output_file=output_file,
        target_mb=199.0, # Στόχος μεγέθους σε MB
        tolerance_mb=1.0,  # Ανοχή σε MB
        max_iter=10 # Μέγιστος αριθμός επαναλήψεων
    )
    print("Done!")
