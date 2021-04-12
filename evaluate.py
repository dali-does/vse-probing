import os
import sys
import time
import numpy as np
import argparse 
import torch

#from pycocotools.coco import COCO
from keras.utils import to_categorical

from embeddings import *
from probing_tasks import probe_num_objects, probe_object_categories, probe_tampered_caption

from probing_model import MultiLayerProbingModel, LinearProbingModel

def get_annotations(coco, img_ids):
    categories = coco.getCatIds()
    num_categories = len(categories)

    category_to_categorical = dict(zip(categories,range(num_categories)))

    # numpy array to keep all the embeddings
    annotations = [None]*len(img_ids)*5
    for i in range(len(img_ids)):
        anns = get_probing_task_data(coco, img_ids[i], category_to_categorical)
        for j in range(5):
            annotations[i*5+j] = anns.copy()
    return annotations


def get_probing_task_data(coco, img_id, category_to_categorical):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    num_categories = len(coco.getCatIds())

    instance_categories = [ann['category_id'] for ann in coco.loadAnns(ann_ids)]
    instance_categories = convert_instances(instance_categories, category_to_categorical)

    num_instances = len(instance_categories)

    categorical_instances = np.array(transform_categorical(instance_categories, num_categories))
    binarized_instances = np.array(np.where(categorical_instances, 1.0, 0.0))

    return [num_instances, binarized_instances]

def transform_categorical(instance_categories, num_categories):
    return np.sum(to_categorical(instance_categories,
                                 num_classes=num_categories),axis=0)

def convert_instances(instance_categories, mapping):
    return list(map(lambda x: mapping[x], instance_categories))

class AnnotationData(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, read_alt_caps):
        self.transformed_captions = []
        self.captions = []
        self.coco_ids = []
        self.has_alt_caps = read_alt_caps

        loc = data_path + '/'

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.im_div = 5
        self.length = len(self.images)


        # Captions
        with open(loc+'%s_caps.txt' % data_split, 'r') as cap_file:
            for line in cap_file:
                self.captions.append(str(line.rstrip()))

        if self.has_alt_caps:
            # Generate tampering data indicies
            rng = np.random.default_rng()
            self.tampered_indices = rng.integers(2, size=self.length)
            # Alternative captions
            with open(loc+'%s_alts.txt' % data_split, 'r') as alt_file:
                i = 0
                for line in alt_file:
                    stripped = str(line.rstrip())
                    self.transformed_captions.append(stripped)
                    if len(stripped) < 1:
                        self.tampered_indices[i] = 0
                    i += 1

        with open(loc+'%s.txt'%data_split, 'r') as coco_file:
            for line in coco_file:
                stripped = line.strip().replace('2014_','')
                img_id=int(''.join(filter(str.isdigit,
                                          stripped)))
                self.coco_ids.append(img_id)

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        # handle the image redundancy
        img_id = int(index//self.im_div)
        image = torch.Tensor(self.images[index])

        caption = self.captions[index]
        if self.has_alt_caps:
            if self.tampered_indices[index]:
                caption = self.transformed_captions[index]

        # Convert caption (string) to word ids.
        return image, caption, index, img_id, self.coco_ids[img_id]

    def get_img_ids(self):
        return self.coco_ids

    def get_tampering_indices(self):
        return self.tampered_indices


def build_annotations(annotation_path, data_path, annotation_split='train2014',
            split_name='train', dataset='coco_precomp', read_alt_caps=True):
    coco_instances = get_coco(annotation_path, annotation_split)
    annotation_data = read_annotations(data_path, dataset, split_name, read_alt_caps)
    tampering_indices = annotation_data.get_tampering_indices()

    annotations = get_annotations(coco_instances, annotation_data.get_img_ids())
    [annotations[i].append(tampering_indices[i]) for i in range(len(annotations))]

    return annotation_data, annotations

def read_annotations(data_path, data_name, split, read_alt_caps):
    dpath = os.path.join(data_path, data_name)
    return AnnotationData(dpath, split, read_alt_caps)

def get_coco(path, split):
    instanceFile = '{}/annotations/instances_{}.json'.format(path, split)
    return COCO(instanceFile)

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def get_parser_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vsemodel', help="The visual-semantic embedding model to probe.", choices=['vsepp', 'vsec', 'hal'])
    parser.add_argument('--unimodel', help="The unimodal embedding model to probe.", choices=['bert', 'gpt2'])
    parser.add_argument('--annotation_path', help="Path to MSCOCO annotations.", default='.')
    parser.add_argument('--data_path', help="Path to the raw MSCOCO data.",default='.')
    parser.add_argument('--split', help="Which MSCOCO datasplit to use.", choices=['train2014', 'val2014'], default='train2014')
    parser.add_argument('--vse_model_path', help="Path to pretrained visual-semantic embedding model.", default='runs/coco_vse++/model_best.pth.tar')
    parser.add_argument('--result_file', help="File to store probing results.", default='out.res')
    parser.add_argument('--task', default='objcat', help="The probing task to execute.", choices=['objcat', 'numobj', 'semcong'])
    parser.add_argument('--seed', default=1974, help="The seed used for the Numpy RNG.")
    parser.add_argument('--probe', help="Which probing model to use.", default="linear", choices=['mlp', 'linear'])

    return parser.parse_args()

if __name__=="__main__":

    opts = get_parser_options()

    result_file = opts.resultfile
    result_file = open(result_file, 'a+')


    # Ugly quick-fix: The models have slightly different implementations (or at least VSE_C)
    if opts.vsemodel == 'vsepp':
        from vsepp.model import VSE, order_sim
        from vsepp.vocab import Vocabulary
    elif opts.vsemodel == 'vsec':
        from VSE_C.model import VSE
        from VSE_C.vocab import Vocabulary
    elif opts.vsemodel == 'hal':
        from hal.model import VSE
        from hal.vocab import Vocabulary
    else:
        print("Model not recognised: ",opts.vsemodel)
        sys.exit(1)


    # Ugly quick-fix to preserve function naming in prints
    def compute_vse_embeddings_adapter(annotation_path):
        return compute_vse_embeddings(VSE, opts.vse_model_path, annotation_path)

    models = [compute_vse_embeddings_adapter]

    if opts.unimodel is not None:
        if opts.unimodel == 'bert':
            models.append(compute_bert_embeddings)
        if opts.unimodel == 'gpt2':
            models.append(compute_gpt_embeddings)


    str_to_task = {
            'objcat': probe_object_categories,
            'numobj': probe_num_objects,
            'semcong': probe_tampered_caption
            }

 
    str_to_probe = {
            'mlp': MultiLayerProbingModel,
            'linear': LinearProbingModel,
            }

    # Previously, many tasks could be performed after each other, but to
    # insure there is no interference we only do them one at a time instead.
    # This is less efficient as the embeddings needs to be computed again.
    probing_tasks = str_to_task[opts.task]

    probing_model = str_to_probe[opts.probe]


    start = time.time()

    train_set_annotation, annotations_train = build_annotations(opts.annotation_path,
                                                                opts.data_path)
    test_set_annotation, annotations_test = build_annotations(opts.annotation_path,
                                                              opts.data_path, 'val2014', 'test')
    print("Building annotations: ", time.time()-start)

    np.random.seed(opts.seed)
    train_indices, val_indices = split_indices(len(annotations_train),val_pct=0.2)


    merge_funcs = [average_embeddings, concatenate_embeddings]

    start = time.time()

    print("Computed embeddings: ", time.time()-start)

    results = []

    num_repetitions = 5
    np.set_printoptions(precision=3)

    # This follows the previous form of doing multiple tasks at once.
    for embedding_model in models:
        print("Probing...", embedding_model.__name__)
        embs = compute_embeddings(embedding_model, train_set_annotation,
                                  test_set_annotation, merge_funcs)
        model_results = []
        for probing_task in probing_tasks:
            print("Probing [%s]"%probing_task.__name__)
            task_results = []
            for emb_pair in embs:
                reps = []
                for k in range(num_repetitions):
                    start = time.time()
                    m, res = probing_task(probing_model, emb_pair[0], annotations_train, emb_pair[1],
                                         annotations_test, train_indices, val_indices)
                    reps.append((res[2], np.array(res[3])))
                    del m
                    print("Probed embedding: ",time.time()-start)
                print("Finished folds")
                reps = np.array(reps)
                stds = np.std(reps[:,0], axis=0)
                reps = np.mean(reps, axis=0)
                task_results.append((reps[0],stds,reps[1]))

                print(probing_task.__name__,embedding_model.__name__, reps[1],
                      file=result_file)
                print("%s on %s: mean: %.3f, std: %.3f"%(probing_task.__name__,embedding_model.__name__, reps[0],
                      stds), file=result_file)
            model_results.append(task_results)
        results.append(model_results)
        del embs

    print("Means and std", file=result_file)
    for i in range(len(models)):
        embedding_model = models[i]
        for j in range(len(probing_tasks)):
            task = probing_tasks[j]
            for mean_metric,std_metric,per_label_acc in results[i][j]:
                print("%s on %s: mean: %.3f, std:%.3f"%(task.__name__,embedding_model.__name__, mean_metric,
                                 std_metric), file=result_file)
