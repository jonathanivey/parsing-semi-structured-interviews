import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
import argparse
import os


def main(transcripts, key_questions, out_path):
    # Read in Key Questions that we will use to split the transcript
    question_df = pd.read_csv(key_questions)
    categories = question_df.iloc[:, 0]
    questions = question_df.iloc[:, 1]

    # load the sentences transformer huggingface model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # process all the .txt files in the transcripts folder
    for filename in os.listdir(transcripts):
        if filename.endswith('.txt'):
            # do preprocessing, find exact question wordings, label and output each cleaned transcript, and return a list of the sections in the transcript
            print('\nProcessing ' + filename)
            process_transcript(model, transcripts, filename, categories, questions, out_path)


# given a transcript, categories, and questions, categorize it appropriately
def process_transcript(model, transcripts, filename, categories, questions, out_path):
    # read in lines from interview file
    with open(transcripts + '/' + filename, errors='ignore') as f:
        lines = f.readlines()

    # remove all newline characters
    lines = [line.strip() for line in lines if line.strip()]

    # combine all the lines into one document
    doc = " ".join(lines)

    # load spacy's english pipeline
    nlp = spacy.load('en_core_web_sm')

    # use spacy's pipeline to separate the document into sentences
    sentences = [i.text for i in nlp(doc).sents]

    # ensure than sentences are split where the speaker changes position
    temp_list = []
    # for each sentence in the list, split on when the interviewer talks and append them all together
    for sentence in sentences:
        temp_list = temp_list + sentence.split('Interviewer')

    sentences = []
    # for each sentence in the list, split on when the participant talks and append them all together
    for sentence in temp_list:
        sentences = sentences + sentence.split('Participant')

    # remove empty items from the list
    sentences = [item for item in sentences if item]

    # find the exact wording used in the transcript for each of the relevant questions
    exact_questions, question_scores = extract_question(model, questions, sentences)

    # read in the transcript again
    with open(transcripts + '/' + filename, errors='ignore') as f:
        transcript = f.read()

    # remove unnecessary newlines
    transcript = transcript.replace('\n\n', '%!%')
    transcript = transcript.replace('\n', ' ')
    transcript = transcript.replace('%!%', '\n')


    # Find each question and insert a category heading at a natural break
    for i, question in enumerate(exact_questions):

        # if the question we find isn't at least 75% similar, don't add the label
        if question_scores[i] < .75:
            print('\nError finding question for ' + categories[i])
            print('\n\tOriginal question wording:\n\t' + questions[i])
            print('\n\tMost similar sentence we found:\n\t' + question)
        else:
            # find the index of the first character of the question
            r_bound = transcript.find(question)

            # if we fail to find a question, then throw an error.
            if r_bound == -1:
                print('Error trying to find beginning question for ' + categories[i])
                print('\n\tFailed to find this question:\n\t' + question)
            else:
                # find the first time that the interviewer speaks before the question
                location = transcript.rfind('Interviewer', 0, r_bound)

                # insert the name of each category at the relevant location in the transcript
                transcript = transcript[:location] + '\n\n[' + categories[i] + ']\n\n' + transcript[location:]

    # add intro tag to beginning of transcript
    transcript = '\n\n[Intro]\n\n' + transcript
    
    # write to a new transcript file
    with open(out_path + '/' + filename, 'w') as f:
        f.write(transcript)



# given a list of interview questions and the preprocessed sentences from a transcript
# find the exact wording of each of those questions and output as a list
def extract_question(model, questions, sentences):

    # compute embeddings for both questions in the script and sentences in the interview
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # compute cosine-similarities
    cosine_scores = util.cos_sim(question_embeddings, sentence_embeddings)

    # create a list of the sentences with the maximum similarity to each of the script's questiosn
    exact_questions = []
    question_scores = []

    for i in range(len(cosine_scores)):
        best_sentence = sentences[torch.argmax(cosine_scores[i])]
        exact_questions.append(best_sentence.strip())
        question_scores.append(torch.max(cosine_scores[i]))

    # return a list of the exact wording for each of the defined questions

    return exact_questions, question_scores
     



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Automatically code interview transcripts')
    parser.add_argument('--transcripts', metavar='path', required=True,
                        help='path to a folder with a set of uncoded interview transcripts in the .txt format.')
    parser.add_argument('--key_questions', metavar='path', required=True,
                        help='path to a .csv file that has two columns. The first column is labeled "Category" and contains the name of each of the sections of interview questions in the order in which they are asked. The second column is labeled "First Question" and contains the first question that is asked in each of the sections of the interview.')
    parser.add_argument('--out_path', metavar='path', required=True,
                        help='path to a folder where you want the newly coded files to be saved.')

    args = parser.parse_args()
    main(transcripts=args.transcripts, key_questions = args.key_questions, out_path=args.out_path)