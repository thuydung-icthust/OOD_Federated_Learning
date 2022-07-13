import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from helper import Helper
import random
import logging

from models.word_model import RNNModel
from text_load import *

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.cuda()

def poison_dataset(params, data_source, dictionary, poisoning_prob=1.0):
    poisoned_tensors = list()

    for sentence in params['poison_sentences']:
        sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                        len(x) > 1 and dictionary.word2idx.get(x, False)]
        sen_tensor = torch.LongTensor(sentence_ids)
        len_t = len(sentence_ids)

        poisoned_tensors.append((sen_tensor, len_t))

    ## just to be on a safe side and not overflow
    no_occurences = (data_source.shape[0] // (params['bptt']))
    logger.info("CCCCCCCCCCCC: ")
    logger.info(len(params['poison_sentences']))
    logger.info(no_occurences)

    for i in range(1, no_occurences + 1):
        if random.random() <= poisoning_prob:
            # if i>=len(self.params['poison_sentences']):
            pos = i % len(params['poison_sentences'])
            sen_tensor, len_t = poisoned_tensors[pos]

            position = min(i * (params['bptt']), data_source.shape[0] - 1)
            data_source[position + 1 - len_t: position + 1, :] = \
                sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

    logger.info(f'Dataset size: {data_source.shape} ')
    return data_source


def load_poisoning_data():
    ### DATA PART
    corpus = None
    default_params = {
        'size_of_secret_dataset': 640,
        'poisoning_per_batch': 1,
        'bptt': 64,
        'word_dictionary_path': 'data/reddit/50k_word_dictionary.pt',
        'number_of_total_participants': 80000,
        'data_folder': 'data/reddit/',
        'recreate_dataset': False,
        'is_poison': True,
        'number_of_adversaries': 1,
        'test_batch_size': 10,
        'batch_size': 20,
        'poisoning': 1.0,
        'poison_sentences': ['pasta from Astoria tastes delicious'],
        # is_poison: true
        # baseline: false
        # random_compromise: false
    }
    logger.info('Loading data')
    #### check the consistency of # of batches and size of dataset for poisoning
    if default_params['size_of_secret_dataset'] % (default_params['bptt']) != 0:
        raise ValueError(f"Please choose size of secret dataset "
                            f"divisible by {default_params['bptt'] }")
    dictionary_pt_file = default_params['word_dictionary_path']
    print(f"dictionary_pt_file: {dictionary_pt_file}")
    dictionary = torch.load(dictionary_pt_file)
    corpus_file_name = f"{default_params['data_folder']}" \
                        f"corpus_{default_params['number_of_total_participants']}.pt.tar"
    if default_params['recreate_dataset']:
        corpus = Corpus(default_params, dictionary=dictionary,
                                is_poison=default_params['is_poison'])
        torch.save(corpus, corpus_file_name)
    else:
        # Load directly from saved data.
        print(f"corpus_file_name: {corpus_file_name}")
        corpus = torch.load(corpus_file_name)
    logger.info('Loading data. Completed.')
    if default_params['is_poison']:
        default_params['adversary_list'] = [POISONED_PARTICIPANT_POS] + \
                                            random.sample(
                                            range(default_params['number_of_total_participants']),
                                            default_params['number_of_adversaries'] - 1)
        logger.info(f"Poisoned following participants: {len(default_params['adversary_list'])}")
    else:
        default_params['adversary_list'] = list()
    ### PARSE DATA
    eval_batch_size = default_params['test_batch_size']
    local_train_data = [batchify(data_chunk, default_params['batch_size']) for data_chunk in
                        corpus.train]
    num_dps = [len(local_data) for local_data in local_train_data]
    test_data = batchify(corpus.test, eval_batch_size)
    num_poisoned_data = default_params['size_of_secret_dataset'] * default_params['batch_size']
    if default_params['is_poison']:
        data_size = test_data.size(0) // default_params['bptt']
        test_data_sliced = test_data.clone()[:data_size * default_params['bptt']]
        test_data_poison = poison_dataset(params=default_params, data_source=test_data_sliced, dictionary=dictionary, poisoning_prob=1.0)
        poisoned_data = batchify(
            corpus.load_poison_data(number_of_words=default_params['size_of_secret_dataset'] *
                                                            default_params['batch_size']),
            default_params['batch_size'])
        poisoned_data_for_train = poison_dataset(params=default_params, data_source=poisoned_data, dictionary=dictionary,
                                                            poisoning_prob=default_params[
                                                                'poisoning'])

    n_tokens = len(corpus.dictionary)
    return n_tokens, corpus, num_poisoned_data, poisoned_data_for_train, test_data_poison, local_train_data, num_dps, test_data
    


def create_model(current_time, n_tokens, corpus, device='cuda'):
    model_params = {
        'emsize': 200,
        'nhid': 200,
        'nlayers': 2,
        'dropout': 0.2,
        'tied': True,
        'bptt': 64,
        'clip': 0.25,
        'seed': 1,
    }
    n_tokens = len(corpus.dictionary)
    local_model = RNNModel(name='Local_Model', created_time=current_time,
                            rnn_type='LSTM', ntoken=n_tokens,
                            ninp=model_params['emsize'], nhid=model_params['nhid'],
                            nlayers=model_params['nlayers'],
                            dropout=model_params['dropout'], tie_weights=model_params['tied'])
    local_model = local_model.to(device)
    return local_model
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def test_poison(params, corpus, epoch, data_source, criterion,
                model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = params['test_batch_size']

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    data_iterator = range(0, data_source.size(0) - 1, params['bptt'])
    dataset_size = len(data_source)

    for batch_id, batch in enumerate(data_iterator):
        # data, targets = helper.get_batch(data_source, batch, evaluation=True)
        seq_len = min(params['bptt'], len(data_source) - 1 - batch)
        data = data_source[batch:batch + seq_len]
        targets = data_source[batch + 1:batch + 1 + seq_len].view(-1)
        
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
        hidden = repackage_hidden(hidden)

        pred = output_flat.data.max(1)[1][-batch_size:]


        correct_output = targets.data[-batch_size:]
        correct += pred.eq(correct_output).sum()
        total_test_words += batch_size


    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss.item() / dataset_size

    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
    # if visualize:
    #     model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
    #                     eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return total_l, acc
