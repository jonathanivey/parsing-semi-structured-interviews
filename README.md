# Parsing Semi-Structured Interviews

## Environment
This project has the following dependencies:
```
numpy==1.24.2
pandas==1.5.3
sentence_transformers==2.2.2
spacy==3.5.4
torch==2.0.1
```


## Function
Using the `main.py` file you can automatically insert section labels for interview transcripts.


## How to Use
### 1. Convert your transcripts to .txt files

You can use `pypandoc` for this as shown below:

```python
pypandoc.convert_file(docxFilename, 'plain', outputfile=txtFilename)
```

### 2. Create input and output folders
All of your transcripts should be in one folder, and you should have a separate empty folder to store the labeled transcripts in. 

### 3. Create a .csv of all your sections
The first column  should contain section names and have the header "Category".

The second column should contain the first question asked within each section, and should have the header "First Question".

### 4. Run the script

Use the following command:
```
python interview-coder.py --transcripts input_folder --key_questions sections.csv --out_path output_folder
```

Where...
- `input_folder` contains your transcripts as .txt files.
- `sections.csv` is the .csv created in step 2.
- `output_folder` is an empty folder where you want to store the labeled transcripts.

**Note: The `process_transcript` function makes modifications to the text files based on the transcripts that were used in testing. You may need to modify it to better handle your own transcripts.**
