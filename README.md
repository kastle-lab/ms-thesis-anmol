# knowledge-extraction-methods-exploration

## Repository Structure

```
knowledge-extraction-methods-exploration/
├── csv/                              # Modular representation of the schema as triples in CSVs (the names of the contained files represent the modules)
├── extracted-knowledge/              # The cleaned data-populated triples by LLMs
├── .gitignore
├── Automated-RAP-gemini.py           # Gemini-specific code for LLM pipeline to generate data-populated triples
├── Automated-RAP.ipynb               # Code for LLM pipeline to generate data-populated triples
├── L156_S2_Roy_2007.xml              # Gold standard XML used as a means of comparison
├── LICENSE
├── README.md
├── knowledge-extraction-cleaning.py  # Code to clean the data-populated triples
├── pub-extraction-eval.py            # Code to evaluate cleaned data-populated triples against publication and gold standard XML
├── publication.md                    # Publication converted into Markdown
├── triples.csv                       # Representation of the schema as triples in a CSV
```

The items contained in `extracted-knowledge/` have the following naming convention: `<model_name>-<boolean>-<boolean>-<boolean>-<boolean>-knowledge-extraction` with no extension for directories and `.csv` for files. The four booleans represent the following configurations:

1. Schema Provision (Complete = True, Modules = False)  
2. Schema Pruning (Yes = True, No = False)  
3. Literature Pruning (Yes = True, No = False)  
4. Example Provision (Yes = True, No = False)
