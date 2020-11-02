import os
import time
import numpy as np
import argparse 
import torch

from probing_tasks import probe_num_objects, probe_object_categories, probe_tampered_caption

from vsepp.model import VSE
from vsepp.vocab import Vocabulary

#from embeddings import average_embeddings, concatenate_embeddings
from embeddings import *

from pycocotools.coco import COCO

from keras.utils import to_categorical

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
    annIds = coco.getAnnIds(imgIds=img_id)
    num_categories = len(coco.getCatIds())

    instance_categories = [ann['category_id'] for ann in coco.loadAnns(annIds)]
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
    def __init__(self, data_path, data_split):
        self.transformed_captions = []
        self.captions = []
        self.coco_ids = []

        loc = data_path + '/'

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.im_div = 5
        self.length = len(self.images)

        # Generate tampering data indicies
        rg = np.random.default_rng()
        self.tampered_indices = rg.integers(2, size=self.length)

        # Captions
        with open(loc+'%s_caps.txt' % data_split, 'r') as f:
            for line in f:
                self.captions.append(str(line.rstrip()))

        # Alternative captions
        with open(loc+'%s_alts.txt' % data_split, 'r') as f:
            i = 0
            for line in f:
                stripped = str(line.rstrip())
                self.transformed_captions.append(stripped)
                if len(stripped) < 1:
                    self.tampered_indices[i] = 0
                i += 1

        with open(loc+'%s.txt'%data_split, 'r') as f:
            for line in f:
                stripped = line.strip().replace('2014_','')
                img_id=int(''.join(filter(str.isdigit,
                                          stripped)))
                self.coco_ids.append(img_id)

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        # handle the image redundancy
        img_id = int(index//self.im_div)
        image = self.images[index]
        #image = image.dot([0.2989,0.5870,0.1140])
        #image = np.min(image, 255).astype(np.uint8)
        image = torch.Tensor(image)

        caption = self.captions[index]
        if self.tampered_indices[index]:
            caption = self.transformed_captions[index]

        # Convert caption (string) to word ids.
        return image, caption, index, img_id, self.coco_ids[img_id]

    def get_img_ids(self):
        return self.coco_ids

    def get_tampering_indices(self):
        return self.tampered_indices


def build_annotations(annotation_path, data_path, annotation_split='train2014',
            split_name='train', dataset='coco_precomp'):
    coco_instances = get_coco(annotation_path, annotation_split)
    annotation_data = read_annotations(data_path,
                                                  dataset,split_name)
    tampering_indices = annotation_data.get_tampering_indices()

    annotations = get_annotations(coco_instances, annotation_data.get_img_ids())
    [annotations[i].append(tampering_indices[i]) for i in range(len(annotations))]

    return annotation_data, annotations

def read_annotations(data_path, data_name, split):
    dpath = os.path.join(data_path, data_name)
    return AnnotationData(dpath, split)

def get_coco(path, split):
    instanceFile = '{}/annotations/instances_{}.json'.format(path, split)
    return COCO(instanceFile)

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def get_parser_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='vsepp')
    parser.add_argument('--annotation_path',default='.')
    parser.add_argument('--data_path',default='.')
    parser.add_argument('--split',default='train2014')
    parser.add_argument('--model_path',default='runs/coco_vse++/model_best.pth.tar')
    parser.add_argument('--embedding_path',default=None)
    parser.add_argument('--resultfile', default='out.res')

    return parser.parse_args()



if __name__=="__main__":

    opts = get_parser_options()

    result_file = opts.resultfile
    result_file = open(result_file, 'a+')


    # Ugly quick-fix: The models have slightly different implementations (or at least VSE_C)
    if opts.model == 'vsepp':
        from vsepp.model import VSE, order_sim
        from vsepp.vocab import Vocabulary
        print("here1")
    elif opts.model == 'vsec':
        print("here2")
        from VSE_C.model import VSE
        from VSE_C.vocab import Vocabulary
    elif opts.model == 'hal':
        from hal.model import VSE
        from hal.vocab import Vocabulary
    else:
        print("Model not recognised: ",opts.model)
        sys.exit(1)



    start = time.time()

    train_set_annotation, annotations_train = build_annotations(opts.annotation_path,
                                                                opts.data_path)
    test_set_annotation, annotations_test = build_annotations(opts.annotation_path,
                                                              opts.data_path, 'val2014', 'test')
    print("Building annotations: ", time.time()-start)

    np.random.seed(1974)
    train_indices, val_indices = split_indices(len(annotations_train),val_pct=0.2)

    probing_tasks = [probe_num_objects, probe_object_categories,
                     probe_tampered_caption]

    # Ugly quick-fix to preserve function naming in prints
    def compute_vse_embeddings_adapter(annotation_path):
        return compute_vse_embeddings(VSE, opts.model_path, annotation_path)

    models = [compute_vse_embeddings_adapter, compute_bert_embeddings, compute_gpt_embeddings]

    merge_funcs = [average_embeddings, concatenate_embeddings]

    start = time.time()

    print("Computed embeddings: ", time.time()-start)

    results = []

    num_repetitions = 5
    np.set_printoptions(precision=3)

    for model in models:
        print("Probing...", model.__name__)
        embs = compute_embeddings(model, train_set_annotation,
                                  test_set_annotation, merge_funcs)
        model_results = []
        for probing_task in probing_tasks:
            print("Probing [%s]"%probing_task.__name__)
            task_results = []
            for emb_pair in embs:
                reps = []
                for k in range(num_repetitions):
                    start = time.time()
                    m, res = probing_task(emb_pair[0], annotations_train, emb_pair[1],
                                         annotations_test, train_indices, val_indices)
                    reps.append((res[2], np.array(res[3])))
                    del m
                    print("Probed embedding: ",time.time()-start)
                print("Finished folds")
                reps = np.array(reps)
                stds = np.std(reps[:,0], axis=0)
                reps = np.mean(reps, axis=0)
                task_results.append((reps[0],stds,reps[1]))

                print(probing_task.__name__,model.__name__, reps[1],
                      file=result_file)
                print("%s on %s: mean: %.3f, std: %.3f"%(probing_task.__name__,model.__name__, reps[0],
                      stds), file=result_file)
            model_results.append(task_results)
        results.append(model_results)
        del embs

    print("Means and std", file=result_file)
    for i in range(len(models)):
        model = models[i]
        for j in range(len(probing_tasks)):
            task = probing_tasks[j]
            for mean_metric,std_metric,per_label_acc in results[i][j]:
                print("%s on %s: mean: %.3f, std:%.3f"%(task.__name__,model.__name__, mean_metric,
                                 std_metric), file=result_file)
