import pandas as pd
import torch
import os

from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from credit_scoring.utils import read_parquet_dataset_from_local
from credit_scoring.dataset_preprocessing_utils import features
from collections import defaultdict

from credit_scoring.data_generators import batches_generator
from credit_scoring.pytorch_training import train_epoch, eval_model, inference
from credit_scoring.training_aux import EarlyStopping

from credit_scoring.data_preprocess import create_buckets_from_credits
from credit_scoring.model import CreditsGRU, CreditsLSTM
from credit_scoring.ne_encoder import NpEncoder
from credit_scoring.get_device import get_device

TRAIN_DATA_PATH = "data_MliF/train_data/"
TEST_DATA_PATH = "data_MliF/test_data/"
TRAIN_TARGET_PATH = "data_MliF/train_target.csv"

TRAIN_BUCKETS_PATH = "data_MliF/train_buckets_rnn"
VAL_BUCKETS_PATH = "data_MliF/val_buckets_rnn"
TEST_BUCKETS_PATH = "data_MliF/test_buckets_rnn"

def compute_embed_dim(n_cat: int) -> int:
    return min(600, round(1.6 * n_cat**0.56))

def data_preprocess():
    train_target = pd.read_csv(TRAIN_TARGET_PATH)
    uniques = defaultdict(set)

    for step in tqdm(range(0, 12, 4),
                                   desc="Count statistics on train data"):
        credits_frame = read_parquet_dataset_from_local(TRAIN_DATA_PATH, step, 4, verbose=True)

        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())

    for step in tqdm(range(0, 2, 2),
                                   desc="Count statistics on test data"):
        credits_frame = read_parquet_dataset_from_local(TEST_DATA_PATH, step, 2, verbose=True)

        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())

    # Задание размерности nn.Embeddings
    uniques = dict(uniques)
    embedding_projections = {feat: (max(uniq)+1, compute_embed_dim(max(uniq)+1)) for feat, uniq in uniques.items()}

    print(sum([embedding_projections[x][1] for x in features]))

    with open("data_MliF/embedding_projections.json", "w") as fp:
        json.dump(embedding_projections, fp, cls=NpEncoder)

    # Для паддинга последовательностей различной длины
    keys_ = list(range(1, 59)) 
    lens_ = list(range(1, 41)) + [45] * 5 + [50] * 5 + [58] * 8
    bucket_info = dict(zip(keys_, lens_))

    with open("data_MliF/bucket_info.json", "w") as fp:
        json.dump(bucket_info, fp, cls=NpEncoder)

    train, val = train_test_split(train_target, random_state=42, test_size=0.1)
    print(f"Train shape: {train.shape}, Val shape: {val.shape}")

    for buckets_path in [TRAIN_BUCKETS_PATH, VAL_BUCKETS_PATH, TEST_BUCKETS_PATH]:
        os.makedirs(buckets_path, exist_ok=True)

    create_buckets_from_credits(TRAIN_DATA_PATH,
                                bucket_info=bucket_info,
                                save_to_path=TRAIN_BUCKETS_PATH,
                                frame_with_ids=train,
                                num_parts_to_preprocess_at_once=4,
                                num_parts_total=12, has_target=True)

    create_buckets_from_credits(TRAIN_DATA_PATH,
                                bucket_info=bucket_info,
                                save_to_path=VAL_BUCKETS_PATH,
                                frame_with_ids=val,
                                num_parts_to_preprocess_at_once=4,
                                num_parts_total=12, has_target=True)

    create_buckets_from_credits(TEST_DATA_PATH,
                                bucket_info=bucket_info,
                                save_to_path=TEST_BUCKETS_PATH, num_parts_to_preprocess_at_once=2,
                                num_parts_total=2)


def train():
    SAVE_PATH_WEIGHT = 'C:/Users/unnamed/ml/alfa/Alfa/weights/BiLSTM_weights_0_0_lr_1e-4_newemb/'
    if not Path.exists(Path(SAVE_PATH_WEIGHT)):
        Path(SAVE_PATH_WEIGHT).mkdir(parents=True, exist_ok=True)

    dataset_train = sorted([os.path.join(TRAIN_BUCKETS_PATH, x) for x in os.listdir(TRAIN_BUCKETS_PATH)])
    dataset_val = sorted([os.path.join(VAL_BUCKETS_PATH, x) for x in os.listdir(VAL_BUCKETS_PATH)])

    es = EarlyStopping(patience=3, mode="max", verbose=True, save_path=os.path.join(SAVE_PATH_WEIGHT, "best_checkpoint.pt"),
                    metric_name="ROC-AUC", save_format="torch")
    num_epochs = 10
    train_batch_size = 128
    val_batch_size = 128

    device = get_device()

    with open("data_MliF/embedding_projections.json") as f_in:
        embedding_projections = json.load(f_in)

    print(sum([embedding_projections[x][1] for x in features]))

    # model = CreditsGRU(features, embedding_projections, bidirectional=True).to(device)
    model = CreditsLSTM(features, embedding_projections, bidirectional=True).to(device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    # print(len(dataset_train))
    # scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=21160)

    for epoch in range(num_epochs):
        train_epoch(model, optimizer, dataset_train, batch_size=train_batch_size,
            shuffle=True, print_loss_every_n_batches=500, device=device)

        val_roc_auc = eval_model(model, dataset_val, batch_size=val_batch_size, device=device)
        es(val_roc_auc, model)

        if es.early_stop:
            break
        torch.save(model.state_dict(), os.path.join(os.path.join(SAVE_PATH_WEIGHT, f"epoch_{epoch+1}_val_{val_roc_auc:.3f}.pt")))

        train_roc_auc = eval_model(model, dataset_train, batch_size=val_batch_size, device=device)




if __name__ == "__main__":
    data_preprocess()
    train()