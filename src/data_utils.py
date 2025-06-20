import json
import random

def generate_dataset(num_rows=2000, symptoms_file="data/raw/symptoms.txt", questions_file="data/raw/questions.txt", output_file="data/processed/output_dataset.jsonl"):
    """
    Generates a JSONL dataset based on symptoms and irrelevant questions.

    Args:
        num_rows (int): The total number of rows (data entries) to generate.
        symptoms_file (str): Path to the file containing symptom descriptions, one per line.
        questions_file (str): Path to the file containing irrelevant questions, one per line.
    """
    try:
        # Load symptoms from the symptoms.txt file
        with open(symptoms_file, 'r', encoding='utf-8') as f:
            symptoms = [line.strip() for line in f if line.strip()]
        
        if not symptoms:
            print(f"Error: No symptoms found in '{symptoms_file}'. Please ensure the file is not empty.")
            return

        # Load questions from the questions.txt file
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        if not questions:
            print(f"Error: No questions found in '{questions_file}'. Please ensure the file is not empty.")
            return
        if len(questions) < 3:
            print(f"Warning: Not enough unique questions ({len(questions)}) for selecting 3 per entry. Some questions may be repeated within an entry.")
            # If there are fewer than 3 questions, repeat them to ensure we can always pick 3
            while len(questions) < 3:
                questions.extend(questions) # Simple repetition


        dataset = []
        instruction_text = "Ask relevant follow-up questions based on the patient's symptom description."

        # Cycle through symptoms to ensure we reach num_rows
        symptom_index = 0
        for i in range(num_rows):
            current_symptom = symptoms[symptom_index % len(symptoms)]
            symptom_index += 1

            # Select 3 unique random questions for the output
            # random.sample ensures unique elements are chosen if k <= len(population)
            selected_questions = random.sample(questions, 3)
            output_questions = "\n- " + "\n- ".join(selected_questions)

            entry = {
                "instruction": instruction_text,
                "input": current_symptom,
                "output": output_questions
            }
            dataset.append(entry)

        # Save the dataset to the specified output file in JSONL format
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in dataset:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Dataset successfully saved to '{output_file}' with {num_rows} rows.")


    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# To run this script:
# 1. Save the symptom data you provided into a file named 'symptoms.txt'.
# 2. Save the question data you provided into a file named 'questions.txt'.
# 3. Save this Python code into a file named 'generate_dataset.py'.
# 4. Run the script from your terminal: python generate_dataset.py
# The output will be printed to your console. You can redirect it to a file:
# python generate_dataset.py > output_dataset.jsonl

if __name__ == "__main__":
    # The symptom and question data from your prompt will be manually saved into files.
    # The script will then read them.
    generate_dataset(num_rows=2000)