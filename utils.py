import os

def read_file(path, data_root=None):
    f = open(path, 'r')
    contents = f.readlines()
    samples = []
    for cnt in contents:
        cnt = cnt.rstrip()
        path, lab = cnt.split(',')
        if data_root is not None:
            path = os.path.join(data_root, path)
        lab = int(lab)
        tup = (path, lab)
        samples.append(tup)

    f.close()
    return samples

def form_samples_classwise(samples, num_classes):
    samples_cl = []
    for cl in range(num_classes):
        samples_cl.append([])

    for sample in samples:
        class_id = sample[1]
        samples_cl[class_id].append(sample)
    return samples_cl

