import os, json, pickle


def save_pkl(d, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(d, f)


def load_pkl(fpath):
    try:
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    except:
        return {}


task_name = "test"
batch_size = 1000
progs_pkl = "/artifact/datasets/test_only/programs_g8_new_13w.pkl"
labels_pkl = "/artifact/datasets/test_only/labels_g8_new_13w.pkl"

progs = load_pkl(progs_pkl)
labels = load_pkl(labels_pkl)
sigs = list(progs.keys())
assert len(sigs) == len(labels)

print(f"Loaded {len(progs)} programs and {len(labels)} labels")


os.makedirs(f"/artifact/datasets/test_only/{task_name}", exist_ok=True)
for i in range(0, len(progs), batch_size):
    if i >= 10*batch_size:
        break
    batch_progs = {}
    batch_labels = {}
    for j in range(i, i+batch_size):
        batch_progs[sigs[j]] = progs[sigs[j]]
        batch_labels[sigs[j]] = labels[sigs[j]]
    print(f"Processing batch {i//batch_size + 1} of {len(progs)//batch_size}")
    print(f"Batch size: {len(batch_progs)}")
    print(f"Batch labels: {len(batch_labels)}")
    save_pkl(batch_progs, f"/artifact/datasets/test_only/{task_name}/progs_batch_{i//batch_size + 1}.pkl")
    save_pkl(batch_labels, f"/artifact/datasets/test_only/{task_name}/labels_batch_{i//batch_size + 1}.pkl")

print('Done')