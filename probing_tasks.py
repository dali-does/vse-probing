import time
import torch
import numpy as np
from progress import progressbar

from probing_model import MultiLayerProbingModel as ProbingModel
from torch.utils.data.sampler import SubsetRandomSampler

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def probe_tampered_caption(embs_train, annotations_train, embs_test,
                           annotations_test, train_indices, val_indices):
    embedding_size = len(embs_train[0])
    nr_of_categories = 2

    _,_, tampered_train = zip(*annotations_train)
    _,_, tampered_test = zip(*annotations_test)

    device = get_device()
    train_loader, validation_loader = get_loaders(embs_train, tampered_train,
                                                  train_indices, val_indices,
                                                  device)
    test_loader = get_loader(embs_test, tampered_test, device)

    probing_model = ProbingModel(embedding_size, nr_of_categories)
                                 
    probing_model = to_device(probing_model, device)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(probing_model.parameters(), lr=0.001)
    metric = binary_acc

    epochs = 30

    return fit_model(probing_model, train_loader, validation_loader,
                     test_loader, loss_fn, optimizer, metric, epochs, nr_of_categories)

def filter_object_categories(embs, categories):
    counts = np.array([np.sum(ann) for ann in categories])
    ones = np.where(counts == 1)


    embs_filtered = np.array(embs)
    embs_filtered = embs_filtered[ones]

    categories_filtered = np.array(categories)
    categories_filtered = categories_filtered[ones]
    categories_filtered = np.argmax(categories_filtered, axis=1)

    return embs_filtered, categories_filtered

def probe_object_categories(embs_train, annotations_train, embs_test,
                            annotations_test, train_indices, val_indices):
    embedding_size = len(embs_train[0])
    nr_of_categories = 80

    _, categories_train, _ = zip(*annotations_train)
    _, categories_test, _ = zip(*annotations_test)

    embs_train, categories_train = filter_object_categories(embs_train,
                                                            categories_train)
    embs_test, categories_test = filter_object_categories(embs_test,
                                                          categories_test)
    np.save('categories_filtered_train.npy',categories_train)
    np.save('categories_filtered_test.npy',categories_test)


    train_indices, val_indices = split_indices(len(embs_train),val_pct=0.2)

    device = get_device()
    train_loader, validation_loader = get_loaders(embs_train, categories_train,
                                                  train_indices, val_indices,
                                                  device)
    test_loader = get_loader(embs_test, categories_test, device)

    probing_model = ProbingModel(embedding_size, nr_of_categories)
    probing_model = to_device(probing_model, device)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(probing_model.parameters(), lr=0.0001)
    metric = accuracy

    epochs = 30

    return fit_model(probing_model, train_loader, validation_loader,
                     test_loader, loss_fn, optimizer, metric, epochs, nr_of_categories)

def probe_num_objects(embs_train, annotations_train, embs_test,
                      annotations_test, train_indices, val_indices):
    nr_of_bins = 6

    num_objects_train, _, _ = zip(*annotations_train)
    num_objects_test, _, _ = zip(*annotations_test)

    binned_annotations_train = objects_to_bins(num_objects_train, nr_of_bins)
    binned_annotations_test = objects_to_bins(num_objects_test, nr_of_bins)

    embedding_size = len(embs_train[0])

    device = get_device()
    train_loader, validation_loader = get_loaders(embs_train,
                                                  binned_annotations_train,
                                                  train_indices, val_indices,
                                                  device)
    test_loader = get_loader(embs_test, binned_annotations_test, device)

    probing_model = ProbingModel(embedding_size, nr_of_bins)
    probing_model = to_device(probing_model, device)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(probing_model.parameters(), lr=0.0001)
    metric = accuracy

    epochs = 30

    return fit_model(probing_model, train_loader, validation_loader,
                     test_loader, loss_fn, optimizer, metric, epochs, nr_of_bins)

def get_loader(embs, annotations, device, label_type=torch.LongTensor):
    tensor_x = torch.Tensor(embs) # transform to torch tensor
    tensor_y = torch.Tensor(annotations).type(label_type)


    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset

    loader = torch.utils.data.DataLoader(dataset, batch_size=512,
                                         pin_memory=True, num_workers=2)

    return loader#DeviceDataLoader(loader, device)

def get_loaders(embs, annotations, train_indices, val_indices, device,
                label_type=torch.LongTensor):
    tensor_x = torch.Tensor(embs) # transform to torch tensor
    tensor_y = torch.Tensor(annotations).type(label_type)


    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512,
                                               pin_memory=True,
                                               num_workers=4)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=512,
                                                    pin_memory=True,
                                                    num_workers=2)

    return train_loader, validation_loader


def fit_model(model, train_loader, validation_loader, test_loader, loss_fn, optimizer,
              metric, epochs, num_classes):
    device = get_device()
    for epoch in progressbar(range(epochs)):
        model.train()
        #start = time.time()
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _,loss,_,_ = loss_batch(model, loss_fn, embeddings, labels,
                                  optimizer)
        #print("Training time: ", time.time()-start)
        #start = time.time()
        #result = evaluate(model, loss_fn, validation_loader, num_classes,
        #                  metric,use_confusion_matrix=False)
        #print("Evaluation time: ", time.time()-start)
        #val_loss, total, val_metric, per_label_acc = result
        #if metric is None:
        #    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs,
        #                                               val_loss))
        #else:
        #    print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch, epochs,
        #                                                           val_loss,
        #                                                           metric.__name__,
        #                                                           val_metric))

    print("Evaluating on test loader")
    avg_loss, total, avg_metric, per_label_acc = evaluate(model, loss_fn, test_loader, num_classes, metric, use_confusion_matrix=True)
    return model, (avg_loss, total, avg_metric, per_label_acc)

def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_fn(preds, yb)
    if opt is not None:
        loss.backward()
        opt.step()
    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)
    return preds, loss, len(xb), metric_result

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)

def evaluate(model, loss_fn, valid_dl, num_classes, metric=None, use_confusion_matrix=False):
    if num_classes < 2:
        num_classes += 1
    confusion_matrix = torch.zeros(num_classes, num_classes).int()
    device = get_device()
    with torch.no_grad():
        model.eval()
        y_test = []
        results = []
        outputs = []
        for xb,yb in valid_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            result = loss_batch(model, loss_fn, xb, yb, metric=metric)
            if use_confusion_matrix:
                _, preds = torch.max(result[0], 1)
                for t, p in zip(yb.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            results.append(result)
            #y_test += yb
        s = confusion_matrix.sum(1)
        for i in range(len(s)):
            if not (s[i] > 0):
                s[i] == 1.0 #diag item will be 0 anyway
        s = s.float()
        per_label_acc = confusion_matrix.diag().float()/s
        data_size = confusion_matrix.sum()
        if use_confusion_matrix:
            print(confusion_matrix)
            print("Per-label accuracy: ",per_label_acc)
            print("Total accuracy: ",(confusion_matrix.diag().sum().float()/data_size).item())
        preds, losses, nums, metrics = zip(*results)

        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums))/total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics,nums))/total
    return avg_loss, total, avg_metric, per_label_acc

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.sum(preds == labels).item()/len(preds)
    return acc

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc.item()

def accuracy_thresh(y_pred, y_true, thresh=0.5):
    data_size = y_pred.size(0)
    acc = np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()
    return acc / data_size

def objects_to_bins(objects, num_bins):
    unique, counts = np.unique(objects, return_counts=True)
    #print("Number of objects in images: ",dict(zip(unique, counts)))

    max_objs = np.max(30)
    bins = np.linspace(0,max_objs,num_bins)
    digits = np.digitize(objects, bins)

    return np.digitize(objects,bins)-1

