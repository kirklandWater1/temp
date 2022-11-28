# -*- coding: utf-8 -*-
import os
import re
import sys
from collections import Counter
import glob
import json
import random
import Levenshtein

from sklearn.model_selection import train_test_split
from sacremoses import MosesTruecaser
from nltk.tokenize import word_tokenize


def split_train_dev(infile, split_size):
    with open(infile, "r", encoding="utf8") as open_file:
        reader = open_file.readlines()
    reader = reader[1:]  # skip header
    print("Data size: ", len(reader))

    # split data into train and test
    train, dev = train_test_split(reader, test_size=split_size, random_state=42)

    print("Train size: ", len(train))
    print("Dev size: ", len(dev))

    with open(infile + ".train.split", "w", encoding="utf8") as open_file1:
        open_file1.write(("sentence" + "\t" + "lable" + "\n"))
        open_file1.writelines(train)

    with open(infile + ".dev.split", "w", encoding="utf8") as open_file2:
        open_file2.write(("sentence" + "\t" + "lable" + "\n"))
        open_file2.writelines(dev)


def tokenzier_truecaser(inputfile, outputfile):
    with open(inputfile, "r", encoding="utf8") as open_file:
        reader = open_file.readlines()
    output_file = open(outputfile, "wb")
    mtr = MosesTruecaser()

    tok_sentances = []
    tok_sentances_with_lable = []
    unique_labels = Counter()

    for line in reader:
        if line != "\n" and line.__contains__("\t"):
            lable, _, sentence = line.split("\t")
            if unique_labels.__contains__(lable) is False:
                unique_labels[lable] = 1
            else:
                unique_labels[lable] += 1
            if re.search('[a-zA-Z]', sentence) is not None:
                tok_sentance = word_tokenize(sentence)
                tok_sentances.append(tok_sentance)
                tok_sentances_with_lable.append(lable + "\t" + " ".join(tok_sentance))

    print("number of unique labels: ", len(unique_labels))

    if os.path.exists(outputfile + ".truecasemodel") is not True:
        mtr.train(tok_sentances, save_to=outputfile + ".truecasemodel")

    my_truecaser = MosesTruecaser(outputfile + ".truecasemodel")

    output_file.write(("sentence" + "\t" + "lable" + "\n").encode("utf8"))

    for line in tok_sentances_with_lable:
        lable, sentence = line.split("\t")
        output_file.write((my_truecaser.truecase(sentence, return_str=True) + "\t" + lable + "\n").encode("utf8"))


def binary_classification(inputfile, outputfile):
    with open(inputfile, "r", encoding="utf8") as open_file:
        reader = open_file.readlines()

    with open(outputfile, "w", encoding="utf8") as open_file:
        reader = reader[1:]  # skip header
        open_file.write("sentence" + "\t" + "lable" + "\n")

        for line in reader:
            if line != "\n" and line.__contains__("\t"):
                sentence, lable = line.split("\t")
                lable = lable.replace("\n", "")
                if lable == "SUD":
                    open_file.write(sentence + "\t" + "0" + "\n")
                elif lable == "HOL":
                    open_file.write(sentence + "\t" + "1" + "\n")


def process_germen_data(raw_input_file):
    tokenzie_file = raw_input_file + ".tok"
    binary_flag = True

    tokenzier_truecaser(raw_input_file, tokenzie_file)
    print("Tokenize and truecase done!")

    if binary_flag is True:
        binary_classification(tokenzie_file, tokenzie_file + ".bin")
        print("Binary  data selection done!")

    print("*********************************")
    if raw_input_file.__contains__("train"):
        print("Splitting data into train and dev...")
        if binary_flag is True:
            split_train_dev(tokenzie_file + ".bin", 0.2)
        else:
            split_train_dev(tokenzie_file, 0.2)
        print("Splitting done!")


def sentence_validattion_check(sentence, last_sentences):
    distance = 20

    if last_sentences[0] !="" and last_sentences[1] !="":
        distances = []

        for last_sentence in last_sentences:
            distances.append(Levenshtein.distance(sentence, last_sentence))
            # distances.append(Levenshtein.distance(sentence[0:20], last_sentence[0:20]))
            # distances.append(Levenshtein.distance(sentence[-20:], last_sentence[-20:]))

        distance = min(distances)


    if sentence != "\n" and \
            sentence != "" and \
            sentence != " " and \
            len(sentence) > 5 and \
            sentence.count(";/") < 1 \
            and sentence.count("&lt;") < 1 \
            and sentence.count("&gt;") < 1 \
            and sentence.__contains__("http://www.") is False \
            and re.search('[a-zA-Z]', sentence) is not None \
            and distance >= 20 \
            and last_sentences[0][0:20] != sentence[0:20] \
            and last_sentences[1][0:20] != sentence[0:20] \
            and sentence[-20:] != last_sentences[0][-20:] \
            and sentence[-20:] != last_sentences[1][-20:]:
        return True

    else:
        return False


def break_into_small_sentences(sentence, trimmed_sentences, random_length_limit):
    # repressively break sentence into smaller sentences if it is longer than 250
    if len(sentence) > random_length_limit:
        # randomly select a symbol to break the sentence into smaller sentences
        symbols = [".", "!", "?", "!", ":", ";", ","]
        symbol = random.choice(symbols)

        last_period_index = sentence.rfind(symbol, 0, random_length_limit)
        if last_period_index != -1:
            trimmed_sentences.append(sentence[0:last_period_index + 1])
            break_into_small_sentences(sentence[last_period_index + 1:], trimmed_sentences, random.randint(50, 250))
        else:
            trimmed_sentences.append(sentence[0:random_length_limit])
            break_into_small_sentences(sentence[random_length_limit:], trimmed_sentences, random.randint(50, 250))

    return trimmed_sentences


def extract_sentences(inputfile, outputfile, lable):
    with open(inputfile, "r", encoding="utf8") as open_file:
        reader = open_file.readlines()

    seed = 42
    random.seed(seed)

    with open(outputfile, "w", encoding="utf8") as open_file:
        open_file.write("sentence" + "\t" + "lable" + "\n")
        last_two_sentences = ["", ""]
        longest_sentence_length = 0
        number_of_extracted_sentences = 0
        for line in reader:
            json_reader_line = json.loads(line)
            text = json_reader_line['text']
            if text != "\n" and text != "" and text != " ":
                sentence = text.strip(" ").strip("\n").replace("\n", "")
                trimmed_sentences = []
                sentence = sentence.replace("&lt;br&gt;", "")
                sentence = sentence.replace("&lt;br /&gt;", "")
                sentence = sentence.replace("&lt;br/&gt;", "")
                sentence = sentence.replace("&lt;br / &gt;", "")

                if len(sentence) > 100:
                    for sub_sentence in break_into_small_sentences(sentence, trimmed_sentences,
                                                                   random.randint(50, 250)):
                        if len(sub_sentence) > longest_sentence_length:
                            longest_sentence_length = len(sub_sentence)

                        if sentence_validattion_check(sub_sentence, last_two_sentences):
                            # remove sentence's starting spaces
                            sub_sentence = sub_sentence.strip(" ").strip("\n")
                            last_two_sentences[0] = last_two_sentences[1]
                            last_two_sentences[1] = sub_sentence
                            # print(sub_sentence + "\t" + lable + "\n")
                            open_file.write(sub_sentence + "\t" + lable + "\n")
                            number_of_extracted_sentences += 1
                else:
                    if sentence_validattion_check(sentence, last_two_sentences):
                        # remove sentence's starting spaces
                        sentence = sentence.strip(" ").strip("\n")
                        # print(sentence + "\t" + lable + "\n")
                        last_two_sentences[0] = last_two_sentences[1]
                        last_two_sentences[1] = sentence
                        open_file.write(sentence + "\t" + lable + "\n")
                        number_of_extracted_sentences += 1
                        if number_of_extracted_sentences >= 160:
                            print(
                                "Warning: number of extracted sentences is 160, which is the same as the number of original sentences!")

        print("longest sentence length: ", longest_sentence_length)
        print("number of original sentences: ", len(reader))
        print("number of extracted sentences: ", number_of_extracted_sentences)


def get_lable(inputfile):
    path = inputfile.split("/")  # split path
    lable = ""
    for folder in path:
        if folder.__contains__("_texts"):
            lable = folder.split("_")[0].upper()
            break
    return lable


def process_italian_data(raw_italian_folder):
    # for sub_folder in glob.glob(raw_italian_folder + "/*"):
    #     sub_folder = sub_folder.replace("\\", "/")
    #     for AA in glob.glob(sub_folder + "/*"):
    #         lable = get_lable(AA)
    #         for file in glob.glob(AA + "/*"):
    #             if file.endswith(".extract") is not True:
    #                 extract_sentences(file, file + ".extract", lable)
    #                 print("Extracting sentences from " + file + " done!")

    extract_sentences(raw_italian_folder + "AA/wiki_02", raw_italian_folder + "AA/wiki_02.extract", "AA")


def main():
    # process_germen_data(raw_input_file = "../data/LSDC/LSDC_1.1.test")
    process_italian_data(raw_italian_folder="../data/wiki_italian_data/eml_texts/")


if __name__ == "__main__":
    main()
