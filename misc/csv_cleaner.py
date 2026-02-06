import pandas as pd

INPUT_CSV = "assignments/zb_assgn/data/gap analysis results/file_content_submission.csv"
OUTPUT_CSV = "assignments/zb_assgn/data/gap analysis results/file_content_submission_top_100.xlsx"

def export_first_x_rows(input_path, output_path, n=100):
    df = pd.read_csv(input_path)
    df.head(n).to_excel(output_path, index=False)

if __name__ == "__main__":
    export_first_x_rows(INPUT_CSV, OUTPUT_CSV)