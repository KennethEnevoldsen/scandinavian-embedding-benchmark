## 

from seb import registries

task_template = """## {task_name}

**Description**: {task_description}
**Main Score**: {task_main_score}
**References**: {task_references}
**Languages**: {task_languages}
**Type**: {task_type}
**Domains**: {task_domains}
**Number of documents**: {task_num_documents}
**Mean Length of documents (characters)**: {task_length_documents:.2f} (std: {task_length_documents_std:.2f})
"""

tasks = registries.tasks.get_all()
for name, getter in tasks.items():
    # print(name)

    task = getter()
    stats = task.get_descriptive_stats()

    print(
        task_template.format(
            task_name=task.name,
            task_description=task.description,
            task_main_score=task.main_score.capitalize(),
            task_references=task.reference,
            task_languages=", ".join(task.languages),
            task_type=task.type,
            task_domains=", ".join(task.domain),
            task_num_documents=stats["num_documents"],
            task_length_documents=stats["mean_document_length"],
            task_length_documents_std=stats["std_document_length"],
        )
    )
