from typing import List
#from sets import Set
import time
import random

from progress import progressbar

from collections import defaultdict

from bertscorer import BertScorer

import stanza

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
import nltk.data

from nltk.corpus import cmudict  # >>> nltk.download('cmudict')
from nltk.stem import WordNetLemmatizer

from pycocotools.coco import COCO

from random import randint
from pathlib import Path

import inflect
from pyinflect import getInflection

import json

import argparse
import torch

dataDir='coco'
dataType='train2017'

#------------------------------------------------------------------------------------#
#
#   Check whether a word starts with a vowel sounding using the CMU dictionary
#
#------------------------------------------------------------------------------------#

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()  # use only the first one

#------------------------------------------------------------------------------------#
#
#   Match capitalization between words
#
#------------------------------------------------------------------------------------#

def mimic_capitalization(word, template):
    if template[0].isupper():
        return word.capitalize()
    else:
        return word


#------------------------------------------------------------------------------------#
#
#   Find position and alternatives for replacement
#
#------------------------------------------------------------------------------------#

def find_pos_and_candidates (sentence, wordseq, lextab):
    i = -1
    pos = 'x'
    for word in sentence.words:
        i = i + 1
        if word.deprel == "root":
            # Do not attempt to replace proper nouns or determiners
            if word.upos == 'NNP' or word.upos == 'DT':
                return (-1, word, pos, [])

            if word.lemma in ["is", "are", "was", "were", "can", "could", "shall", "should", "will", "would", "must"]:
                return (-1, word, pos, [])

            # Only replace nouns with nouns, verbs with verbs, etc
            trans = False
            if word.upos == 'NOUN':
                pos = 'n'
            if word.upos == 'VERB':
                pos = 'v'
                trans = has_object(sentence, word.id)
            if word.upos == 'ADJ':
                pos = 'a'

            # Use lesk to find the most likely synset
            synset = lesk(wordseq, word.lemma, pos)


              # Expand the list of word senses to include hyponyms and hypernyms
            keys = build_keys(synset,pos,trans)
            replacements = lextab[keys[0]]
            if not replacements:
                if (len(keys) > 1):
                    replacements = lextab[keys[1]]

              # Select a lemma from a sufficiently distant synset
            if not replacements:
                print("Found no replacement for the word", word.text, " ", keys)
                return (-1, word, pos, [])
            if replacements == set():
                print("Found no replacement for the word", word.text, " ", keys)
                return (-1, word, pos, [])
            return (i, word, pos, replacements)
    return (-1, sentence.words[0], pos, [])


#------------------------------------------------------------------------------------#
#
#   Generated alternatives
#
#------------------------------------------------------------------------------------#

def generate (N, j, word, pos, wordseq, replacements):

    alternatives = [[] for i in range(N)]

    for i in range(0, N):
        alternatives[i] = wordseq[:]

        assert isinstance(replacements, set)
        rep_lemma, *rep_lemmas = random.sample(replacements,1)
        rep = str(rep_lemma)

        # Inflect to get the right form of the word
        if not (word.xpos == "JJ" or word.xpos == "NN"):
            if not (getInflection(rep_lemma, tag=word.xpos) == None):
                rep, *rep_list = getInflection(rep_lemma, tag=word.xpos)

        # Mimic capitalization and plurality
        rep_words = rep.split("_")
        l = len(rep_words)
        if (pos == 'n'):
            rep_words[l - 1] = mimic_plurality(rep_words[l - 1], word.text)
        rep_words[0] = mimic_capitalization(rep_words[0], word.text)

        # Fix the form of the determinier, i.e., a or an
        last_index = j - 1
        if (last_index >= 0):
            last_word = alternatives[i][last_index]
            if (last_word.lower() == "a" or last_word.lower() == "an"):
                if starts_with_vowel_sound(rep_words[0].lower()):
                    alternatives[i][last_index] = mimic_capitalization("an", last_word)
                else:
                    alternatives[i][last_index] = mimic_capitalization("a", last_word)

        alternatives[i][j] = " ".join(rep_words)
#    for i in range(0, N):
#        print(" ".join(alternatives[i]))
    return alternatives

#------------------------------------------------------------------------------------#
#
#   Transformation function
#
#------------------------------------------------------------------------------------#
class Transformer():
    def __init__(self, lextab):
        self.lextab = lextab

        self.p = inflect.engine()
        self.scorer = BertScorer()

    def transform_captions(self, doc):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        best_strs = [""]*len(doc)
        N = 10
        alternatives = []
        no_replacements = []
        irreplacable_words = []
    
        doc_index = 0 
        # doc really only contains one sentence, but I only know how to access the first element via a for loop
        for sentence in doc:
            # Construct disambiguation sentence
            wordseq = [word.text for word in sentence.words]
    
            (j, word, pos, replacements) = find_pos_and_candidates(sentence,
                                                                   wordseq,
                                                                   self.lextab)
    
            # If replacements have not been found, return original string
            if not replacements:
                print("No replacements for the word ", word.text)
                no_replacements.append(doc_index)
                irreplacable_words.append(word.text)
            else:
                # If replacements have been found,
                alternatives.append(self.generate(N, j, word, pos, wordseq,
                                                  replacements))
            doc_index += 1
        alts_joined = []

        for alt in alternatives:
            alts_joined += [" ".join(a) for a in alt]
        scores = self.scorer.get_scores(alts_joined)

        alt_index = 0
        for i in range(len(doc)):
            if i not in no_replacements:
                best_str = ""
                best_score = float('inf')
                for j in range(0,N):
                    alt = alts_joined[alt_index*N+j]
                    score = scores[alt_index*N+j]

                    if score <= best_score:
                        best_score = score
                        best_str = alt
                best_str = best_str.replace(" .", ".")
                best_str = best_str.replace(" ,", ",")
                best_str = best_str.replace(" ?", "?")
                best_str = best_str.replace(" !", "!")
                best_str = best_str.replace(" :", ":")
                best_str = best_str.replace(" ;", ";")
                best_str = best_str.replace(" - ", "-")
                best_str = best_str.replace(" '", "'")
                best_strs[i] = best_str
                alt_index +=1
            else:
                best_strs[i] = ""
        return best_strs, irreplacable_words

    def transform (self, line):
        doc = self.nlp(line+"This is a test")
    
        best_str = ""
    
        # doc really only contains one sentence, but I only know how to access the first element via a for loop
        for sentence in doc.sentences:
            # Construct disambiguation sentence
            wordseq = [word.text for word in sentence.words]
    
            (j, word, pos, replacements) = find_pos_and_candidates(sentence,
                                                                   wordseq,
                                                                   self.lextab)
    
            # If replacements have not been found, return original string
            if not replacements:
                print("No replacements for the word ", word.text)
                best_str = " ".join(wordseq)
                break
    
            # If replacements have been found,
            N = 10
            alternatives = self.generate(N, j, word, pos, wordseq, replacements)
    
            # Return the highest-scoring alternative
            best_score = 0.0
            for i in range(0,N):
                alt = " ".join(alternatives[i])
                score = self.scorer.get_score(alt)
                if score >= best_score:
                    best_score = score
                    best_str = " ".join(alternatives[i])

        best_str = best_str.replace(" .", ".")
        best_str = best_str.replace(" ,", ",")
        best_str = best_str.replace(" ?", "?")
        best_str = best_str.replace(" !", "!")
        best_str = best_str.replace(" :", ":")
        best_str = best_str.replace(" ;", ";")
        best_str = best_str.replace(" - ", "-")
        best_str = best_str.replace(" '", "'")

        return best_str
    #------------------------------------------------------------------------------------#
    #
    #   Generated alternatives
    #
    #------------------------------------------------------------------------------------#
    def generate (self, N, j, word, pos, wordseq, replacements):

        alternatives = [[] for i in range(N)]

        for i in range(0, N):
            alternatives[i] = wordseq[:]

            assert isinstance(replacements, set)
            rep_lemma, *rep_lemmas = random.sample(replacements,1)
            rep = str(rep_lemma)

            # Inflect to get the right form of the word
            if not (word.xpos == "JJ" or word.xpos == "NN"):
                if not (getInflection(rep_lemma, tag=word.xpos) == None):
                    rep, *rep_list = getInflection(rep_lemma, tag=word.xpos)

            # Mimic capitalization and plurality
            rep_words = rep.split("_")
            l = len(rep_words)
            if (pos == 'n'):
                rep_words[l - 1] = self.mimic_plurality(rep_words[l - 1], word.text)
            rep_words[0] = mimic_capitalization(rep_words[0], word.text)

            # Fix the form of the determinier, i.e., a or an
            last_index = j - 1
            if (last_index >= 0):
                last_word = alternatives[i][last_index]
                if (last_word.lower() == "a" or last_word.lower() == "an"):
                    if starts_with_vowel_sound(rep_words[0].lower()):
                        alternatives[i][last_index] = mimic_capitalization("an", last_word)
                    else:
                        alternatives[i][last_index] = mimic_capitalization("a", last_word)

            alternatives[i][j] = " ".join(rep_words)
        return alternatives

    #------------------------------------------------------------------------------------#
    #
    #   Match capitalization between words
    #
    #------------------------------------------------------------------------------------#

    def mimic_plurality(self, word, template):
        if self.p.singular_noun(template):
            return self.p.plural(word)
        return word

#------------------------------------------------------------------------------------#
#
#   Build key for indexing synsets
#
#------------------------------------------------------------------------------------#
def build_keys (synset, pos, trans):
    # The pos is always defined and will be used as a backup
    keys = []
    # If the synset is defined we can use it to create a better key
    if not synset is None:
        # The prefix of the key is the name of the synset
        key = synset.lexname().lower()
        # If the synset is a verb, we include its frame in the name
        if (synset.name().find(".v.") > 0):
            key + str(trans)
        keys.append(key)
    keys.append(pos)
    return keys


def build_key (synset, pos):

    # The pos is always defined and will be used as a backup
    short_key = pos
    long_key = None

    # If the synset is defined we can use it to create a better key
    if not synset is None:
        # The prefix of the key is the name of the synset
        long_key = synset.lexname().lower()

        # If the synset is a verb, we include its frame in the name
        if (synset.name().find(".v.") > 0):
            long_key = long_key + str(synset.frame_ids())

    return (long_key,short_key)

#------------------------------------------------------------------------------------#
#
#   Translate between pos tags
#
#------------------------------------------------------------------------------------#

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

#------------------------------------------------------------------------------------#
#
#   Create dictionary with lemmas that occur in COCO
#
#------------------------------------------------------------------------------------#


def build_dictionary (caption_path, dict_path, refresh_file = False,
                      batch_size=1024):

    # Create dictionary as set
    dictionary = set([])
    parsed_captions = []
    verbtab = defaultdict(set)
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                               tokenize_pretokenized=True)

    # Check if dictionary already exists and otherwise create it
    dict_file = Path(dict_path)
    # Set refresh file to true to force regeneration of the dictionary
    if not dict_file.is_file() or refresh_file:

        # Create dictionary of lemmas occurring in coco
        lemmatizer = WordNetLemmatizer()
        with open(caption_path) as caption_file:
            caption_data = CaptionDataset(caption_file)
            caption_loader = torch.utils.data.DataLoader(dataset=caption_data,
                                                        batch_size=batch_size,
                                                        num_workers=2,
                                                         pin_memory=True)
            for captions in progressbar(caption_loader,
                                        "Generating replacements (batch size %d): "%batch_size):
                captions = "\n".join(captions)
                (lemmas, parsed) = pre_process(nlp,captions, lemmatizer,
                                                        verbtab)
                parsed_captions += parsed
                dictionary |= set (lemmas)

        # Write dictionary
        f = open(dict_file, "w")
        for word in progressbar(dictionary, "Writing dictionary to file: "):
            f.write(word)
            f.write("\n")
        f.close()
        print("Wrote", len(dictionary), "words to file", dict_file)
    else:
        f = open(dict_file, "r")
        dictionary = set(f.read().split())

        f.close()
        print("Loaded", len(dictionary), "words from file", dict_file)

    return dictionary, parsed_captions, verbtab

#------------------------------------------------------------------------------------#
#
#   Understand if verb is transitive
#
#------------------------------------------------------------------------------------#

def has_object (sentence, id):
    for word in sentence.words:
        if word.deprel == "obj" and word.head == id:
            return True
    return False

def pre_process (nlp, captions, lemmatizer, verbtab):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    doc = nlp(captions)
    lemmas = set([])

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV }

    for sentence in doc.sentences:
        for word in sentence.words:
            if (word.xpos[0] in tag_dict.keys()):
                pos = tag_dict[word.xpos[0]]
                lm = lemmatizer.lemmatize(word.text, pos).lower()
                lemmas.add(lm)

                if pos == wordnet.VERB:
                    trans = has_object(sentence, word.id)
                    verbtab[trans].add(lm)

    return (lemmas, doc.sentences)

#------------------------------------------------------------------------------------#
#
#   Sort wordnet lemmas
#
#------------------------------------------------------------------------------------#

def sort_wordnet_lemmas (dictionary, verbtab):
    lextab = defaultdict(set)
    for syn in wordnet.all_synsets():

        # Get part of speech
        if (syn.name().find(".n.") > -1):
            pos = 'n'
        elif (syn.name().find(".v.") > -1):
            pos = 'v'
        elif (syn.name().find(".a.") > -1):
            pos = 'a'
        elif (syn.name().find(".s.") > -1):
            pos = 's'
        elif (syn.name().find(".r.") > -1):
            pos = 'r'

        # Sort lemmas into groups based on the keys
        # Filter out lemmas that are hard to inflect and use as replacements
        for lemma in syn.lemmas():

            # avoid proper names
            if lemma.name()[0].isupper():
                continue

            # Make sure that the lemma is in COCO
            if not lemma.name() in dictionary:
                 continue

            # Lemmas with underscores are hard to inflect
            if lemma.name().find("_") > -1:
                continue


            keys = []
            if lemma.name() in verbtab[True]:
                keys += build_keys(syn, pos, True)
            if lemma.name() in verbtab[False]:
                keys += build_keys(syn, pos, False)

            for key in keys:
                lextab[key].add(lemma.name().lower())
    return lextab


#------------------------------------------------------------------------------------#
#
#   Clean caption
#
#------------------------------------------------------------------------------------#

def clean (caption):
    caption = caption.replace("/", " or ")
    caption = caption.replace("#", "")
    caption = caption.strip()
    return caption

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, captions_file):

        self.captions = [line.strip() for line in captions_file]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index].strip()
        if '.' not in caption:
            caption += '.'
        return caption

#------------------------------------------------------------------------------------#
#
#   Main function
#
#------------------------------------------------------------------------------------#

if __name__=='__main__':

    stanza.download('en') # download English model

    # Parse arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--datadir', default=None)
    args_parser.add_argument('--out', default=None)
    args_parser.add_argument('--captions', default='dev.txt')
    args_parser.add_argument('--datatype', default='val2014')
    args_parser.add_argument('--generate_dictionary', action='store_true')
    args_parser.add_argument('--dict', default='dict.txt')

    opts = args_parser.parse_args()

    # Overwrite global variables if arguments are given
    if opts.datadir != None:
        dataDir = opts.datadir
        dataType = opts.datatype

    refresh_file=opts.generate_dictionary

    # Select annotation file
    annotations_path = '%s/captions_%s.json' % (dataDir, dataType)
    caption_path = opts.captions

    dict_path = opts.dict
    if opts.dict is None:
        dict_path='%s/dict_%s.txt'%(dataDir,dataType)

    # Build a dicitionary with the lemmas found in the COCO captions
    (dictionary, parsed_captions, verbtab) = build_dictionary(caption_path, dict_path,
                                  refresh_file=refresh_file)
    
    print("Built dictionary")
    # Order the synsets by Wordnet lexical name, e.g. verb.motion
    lextab = sort_wordnet_lemmas(dictionary, verbtab)
    print("Built lextab")
    
    # Seed random values
    random.seed(1972)

    transformer = Transformer(lextab)

    batch_size = 128

    alt_captions = [""]*len(parsed_captions)
    i = 0 
    altFile = '%s/alt_captions_%s.json'%(dataDir,dataType)
    altFile = open(altFile, 'w+')

    for batch_offset in range(0, len(parsed_captions), batch_size):
        batch = parsed_captions[batch_offset:batch_offset+batch_size] # the result might be shorter than batchsize at the end
        
        start = time.time()
        alts,_ = transformer.transform_captions(batch)
        for alt_ind in range(len(alts)):
            alt_captions[i*batch_size+alt_ind] = alts[alt_ind]
        i += 1

    if opts.out is not None:
        with open(opts.out, 'w') as outfile:
            for alt_caption in alt_captions:
                print(alt_caption, file=outfile)
    else:
        print(alt_captions[:10])

    print("Finished computations in [%d] s."%(time.time()-start))
    # known bugs:
        # fucks up - and '
        # inflection sometimes fails
        # replacing has and is can get weird
        # sometimes confuses composite words, e.g., close up becomes smell up
