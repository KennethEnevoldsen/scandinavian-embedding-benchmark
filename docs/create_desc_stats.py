from pathlib import Path

import pandas as pd
from seb import registries
from tqdm import tqdm


def insert_table(file: Path, table: str) -> None:
    # Read the original Markdown file
    with file.open("r") as f:
        content = f.read()

    # Split the content at the start and end placeholders
    start_section, end_section = (
        content.split("<!--START_TABLE-->")[0],
        content.split("<!--END_TABLE-->")[1],
    )

    # Construct the updated content
    updated_content = start_section + "<!--START_TABLE-->\n" + table + "\n<!--END_TABLE-->" + end_section

    # Write the updated content back to the file
    with file.open("w") as f:
        f.write(updated_content)


def create_table() -> pd.DataFrame:
    # Initialize an empty DataFrame
    columns = [
        "Dataset",
        "Description",
        "Main Score",
        "Languages",
        "Type",
        "Domains",
        "Number of Documents",
        "Mean Length of Documents (characters)",
    ]
    data = []

    tasks = registries.tasks.get_all().items()
    tasks = sorted(tasks, key=lambda x: x[0])

    pbar = tqdm(tasks, desc="Creating Table")
    for name, getter in pbar:
        pbar.set_postfix_str(name)
        task = getter()
        stats = task.get_descriptive_stats()

        # Create a row for each task
        row = [
            f"[{task.name}]({task.reference})",  # Dataset with hyperlink
            task.description,  # Description
            task.main_score.capitalize(),  # Main Score
            ", ".join(task.languages),  # Languages
            task.task_type,  # Type
            ", ".join(task.domain),  # Domains
            stats["num_documents"],  # Number of Documents
            f'{stats["mean_document_length"]:.2f} (std: {stats["std_document_length"]:.2f})',  # Mean Length of Documents
        ]
        data.append(row)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=columns)

    return df


def main():
    path = Path(__file__).parent / "datasets.md"

    df = create_table()
    insert_table(path, df.to_markdown(index=False))


if __name__ == "__main__":
    main()
