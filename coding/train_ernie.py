from functools import partial

import paddle
import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, BertTokenizer, BertForTokenClassification
from paddlenlp.metrics import ChunkEvaluator

from data_processing import load_dict, load_dataset, parse_decodes
from model import ErnieCrfForTokenClassification

def convert_to_features(example, tokenizer, label_vocab):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']
    tokenized_input['labels'] = [label_vocab[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']

@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        preds = model(input_ids, seg_ids, lengths=lens)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))
    model.train()
    return precision, recall, f1_score

@paddle.no_grad()
def predict(model, data_loader, ds, label_vocab):
    all_preds = []
    all_lens = []
    for input_ids, seg_ids, lens, labels in data_loader:
        preds = model(input_ids, seg_ids, lengths=lens)
        # Drop CLS prediction
        preds = [pred[1:] for pred in preds.numpy()]
        all_preds.append(preds)
        all_lens.append(lens)
    sentences = [example[0] for example in ds.data]
    results = parse_decodes(sentences, all_preds, all_lens, label_vocab)
    return results


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

from pathlib import Path

def run_ernie():
    train_dataset, dev_dataset, test_dataset = load_dataset(datafiles=(
                                                            ('./dataset/train.jsonl'), 
                                                            ('./dataset/admin.jsonl'),
                                                            ('./dataset/admin.jsonl')))
    
    label_vocab = load_dict("./dataset/label_config.json")
    
    tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
    ernie = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))
    model = ErnieCrfForTokenClassification(ernie)

    # tokenizer = BertTokenizer.from_pretrained('bert-wwm-ext-chinese')
    # model = BertForTokenClassification.from_pretrained("bert-wwm-ext-chinese", num_classes=len(label_vocab))

    trains_func = partial(
        convert_to_features, tokenizer=tokenizer, label_vocab = label_vocab)
    
    train_dataset.map(trains_func)
    dev_dataset.map(trains_func)
    test_dataset.map(trains_func)

    batchify_fn = lambda sample, fn=Tuple(
        Pad(pad_val=tokenizer.pad_token_id, axis=0, dtype='int32'),
        Pad(pad_val=tokenizer.pad_token_type_id, axis=0, dtype='int32'),
        Stack(dtype='int64'),
        Pad(pad_val=label_vocab.get("O", 0), axis=0, dtype='int64')
    ): fn(sample)

    batch = 8
    train_loader = create_dataloader(dataset=train_dataset,
                                    mode='train',
                                    batch_size=batch,
                                    batchify_fn=batchify_fn
                                    )
    dev_loader = create_dataloader(
        dataset=dev_dataset,
        mode='dev',
        batch_size=batch,
        batchify_fn=batchify_fn)

    test_loader = create_dataloader(
        dataset=test_dataset,
        mode='test',
        batch_size=batch,
        batchify_fn=batchify_fn)

    pdparams_path = Path('./bert_pdparams/best_model.pdparams')
    if not pdparams_path.is_file():
        print("训练开始：")
        # Define the model netword and its loss
        
        metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
        # loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)

        # 配置学习率策略，解决loss=nan问题
        optimizer = paddle.optimizer.AdamW(
            learning_rate=2e-5, parameters=model.parameters())
        
        epochs = 100
        step = 0
        best_f1 = 0
        only_save_best = True
        best_epoch = 0
        for epoch in range(0, epochs):
            for input_ids, token_type_ids, lengths, labels in train_loader: # 循环退不出来
                # print("input_ids:{}".format(input_ids))
                loss = model(input_ids, token_type_ids, lengths=lengths, labels=labels)
                # logits = model(input_ids, token_type_ids)
                # loss = paddle.mean(loss_fn(logits, labels))
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                # lr_scheduler.step()
                if (step % 10 == 0):
                    print("[TRAIN] Epoch:%d - Step:%d - Loss:%f" % (epoch, step, avg_loss))
                step += 1
                # break
            precision, recall, f1_score = evaluate(model, metric, dev_loader)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_epoch = epoch
                if only_save_best:
                    paddle.save(model.state_dict(), './bert_pdparams/best_model.pdparams')
                else:
                    paddle.save(model.state_dict(), './bert_pdparams/model_%d.pdparams' % step)
    else:
        print("模型测试。。。")
        load_layer_state_dict = paddle.load('./pdparams/best_model.pdparams')
        model.load_dict(load_layer_state_dict)
        preds = predict(model, test_loader, test_dataset, label_vocab)
        file_path = "ernie_results.txt"
        with open(file_path, "w", encoding="utf8") as fout:
            fout.write("\n".join(preds))
        # Print some examples
        print(
            "The results have been saved in the file: %s, some examples are shown below: "
            % file_path)
        print("\n".join(preds[:10]))




if __name__ == '__main__':
    run_ernie()


