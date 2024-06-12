# Email-Classification-System
An Agile Workflow-based Email Classification System: This repository contains a machine learning project that classifies emails into multiple categories using an agile workflow. The project includes preprocessing, model building, and unit testing, all defined in a YAML workflow.
## Project Description

This project aims to develop an email classifier for a multinational IT company that receives a large number of customer emails. These emails can be classified into different classes on multiple levels. The ultimate goal is to develop a chatbot that can automatically read and respond to customer emails.

## Workflow

The workflow for email classification involves machine translation, noise removal, and model building. The workflow is defined in the `workflow.yml` file.

### Environment Specification

The project uses a specific environment for execution, defined in the `environment` section of the `workflow.yml` file. The environment, named `text-classification`, requires several dependencies, including Python 3.8, numpy 1.21.2, pandas 1.3.3, stanza 1.3.1, transformers 4.12.0, and scikit-learn 0.24.2.

### Workflow Execution

The workflow execution involves loading and preprocessing data from a CSV file. This step is executed by running the `src.py` script.

### Unit Testing

Unit tests are included to ensure the correctness of the machine translation, noise removal, and model building components. The tests are run using pytest, which can be installed with the command `pip install pytest`.

### Execution Instructions

1. Clone the repository.
2. Install the dependencies using the provided environment specification.
3. Execute the workflow steps by running the specified commands.
4. Run unit tests to ensure the correctness of components.

## Author Information

- Name: Naresh Kumar Satish
- Email: nareshkumar.satish30@gmail.com
