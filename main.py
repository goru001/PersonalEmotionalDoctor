from fastai.text import *
import fastText as ft

from config import config
from data import Seq2SeqDataset
from model import Seq2SeqRNN, Seq2SeqAttnRNN

PATH = Path(config.path)

TMP_PATH = PATH/'tmp'
TMP_PATH.mkdir(exist_ok=True)


def toks2ids(tok,pre):
    freq = Counter(p for o in tok for p in o)
    itos = [o for o,c in freq.most_common(40000)]
    itos.insert(0, '_bos_')
    itos.insert(1, '_pad_')
    itos.insert(2, '_eos_')
    itos.insert(3, '_unk')
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    ids = np.array([([stoi[o] for o in p] + [2]) for p in tok])
    np.save(TMP_PATH/f'{pre}_ids.npy', ids)
    pickle.dump(itos, open(TMP_PATH/f'{pre}_itos.pkl', 'wb'))
    return ids,itos,stoi

def load_ids(pre):
    ids = np.load(TMP_PATH/f'{pre}_ids.npy')
    itos = pickle.load(open(TMP_PATH/f'{pre}_itos.pkl', 'rb'))
    stoi = collections.defaultdict(lambda: 3, {v:k for k,v in enumerate(itos)})
    return ids,itos,stoi

def get_vecs(lang, ft_vecs):
    vecd = {w:ft_vecs.get_word_vector(w) for w in ft_vecs.get_words()}
    pickle.dump(vecd, open(PATH/f'wiki.{lang}.pkl','wb'))
    return vecd

def seq2seq_loss(input, target):
    sl,bs = target.size()
    sl_in,bs_in,nc = input.size()
    if sl>sl_in: input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    return F.cross_entropy(input.view(-1,nc), target.view(-1))#, ignore_index=1)

def test(learn, val_dl, en_itos):
    x, y = next(iter(val_dl))
    probs = learn.model(V(x))
    preds = to_np(probs.max(2)[1])
    print(' '.join([en_itos[o] for i in range(x.shape[0]) for o in x[i] if o != 1]))
    print(' '.join([en_itos[o] for i in range(y.shape[0]) for o in y[i] if o != 1]))
    print(' '.join([en_itos[o] for i in range(preds.shape[0]) for o in preds[i] if o != 1]))

def main():
    df = pd.read_csv(PATH/'happiness_unlimited.csv')
    all_conversation = df['conversation']
    # knowledge base
    # with open('/home/gaurav/Downloads/bk/englishAvyaktMurlis.pkl', 'rb') as f:
    #     englishAvyaktMurlis = pickle.load(f)
    # # Tokens should be created for both conversational data and knowledge base, therefore:
    # total_data = []
    # total_data.extend(all_conversation)
    # total_data.extend(englishAvyaktMurlis)
    # Uncomment the following lines if there's change in data
    # tok = Tokenizer.proc_all_mp(partition_by_cores(total_data))
    # pickle.dump(tok, (PATH / 'tok.pkl').open('wb'))
    # tok = pickle.load((PATH / 'tok.pkl').open('rb'))
    # en_ids, en_itos, en_stoi = toks2ids(tok, 'en')

    en_ids, en_itos, en_stoi = load_ids('en')
    print('Sanity check for tokenization', [en_itos[o] for o in en_ids[0]], len(en_itos))

    # en_vecs = ft.load_model(config.fasttext_encoding_path)
    # en_vecd = get_vecs('en', en_vecs)

    en_vecd = pickle.load(open(PATH / 'wiki.en.pkl', 'rb'))

    # ft_words = en_vecs.get_words(include_freq=True)
    # ft_word_dict = {k: v for k, v in zip(*ft_words)}
    # ft_words = sorted(ft_word_dict.keys(), key=lambda x: ft_word_dict[x])
    # print('total fasttext words', len(ft_words))

    dim_en_vec = len(en_vecd[','])

    # en_vecs = np.stack(list(en_vecd.values()))
    # print('Mean and Std Dev. for en_vecs', en_vecs.mean(), en_vecs.std())

    # Preparing model data
    # Tokenized knowledge base
    en_ids_kb = en_ids[len(all_conversation):]
    # Tokenized question answer pairs
    en_ids = en_ids[:len(all_conversation)]
    # TODO: Try with a different percentile later
    enlen_90 = int(np.percentile([len(o) for o in en_ids], 99))
    en_ids_tr = np.array([o[:enlen_90] for o in en_ids])

    # Even rows speaker is Suresh Oberoi (Question), Odd rows speaker is Sister Shivani (Answer)
    en_ids_tr_q = en_ids_tr[0:][::2]  # even
    en_ids_tr_a = en_ids_tr[1:][::2]  # odd

    # TODO: 13-fold CV for the 13 chapters we have
    np.random.seed(42)
    ch_start = int((df[df['chapter_per_conversation'] == 13]).index[0] / 2)
    trn_keep, val_keep = list(range(len(en_ids_tr_q)))[:ch_start], list(range(len(en_ids_tr_q)))[ch_start:]
    en_trn_q, en_trn_a = en_ids_tr_q[trn_keep], en_ids_tr_a[trn_keep]
    en_val_q, en_val_a = en_ids_tr_q[val_keep], en_ids_tr_a[val_keep]
    # len(en_trn_q), len(en_val_q)

    trn_ds = Seq2SeqDataset(en_trn_q, en_trn_a)
    val_ds = Seq2SeqDataset(en_val_q, en_val_a)

    trn_samp = SortishSampler(en_trn_q, key=lambda x: len(en_trn_q[x]), bs=config.batch_size)
    val_samp = SortSampler(en_val_q, key=lambda x: len(en_val_q[x]))

    trn_dl = DataLoader(trn_ds, config.batch_size, transpose=True, transpose_y=True, num_workers=1,
                        pad_idx=1, pre_pad=False, sampler=trn_samp)
    val_dl = DataLoader(val_ds, int(config.batch_size), transpose=True, transpose_y=True, num_workers=1,
                        pad_idx=1, pre_pad=False, sampler=val_samp)
    md = ModelData(PATH, trn_dl, val_dl)

    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    # Normal seq2seq model
    # rnn = Seq2SeqRNN(en_vecd, en_itos, dim_en_vec, en_vecd, en_itos, dim_en_vec, config.gru_hidden_units, enlen_90)
    # learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    # learn.crit = seq2seq_loss

    # With attn
    rnn = Seq2SeqAttnRNN(en_vecd, en_itos, dim_en_vec, en_vecd, en_itos, dim_en_vec, config.gru_hidden_units, enlen_90, knowledge_base_ids=en_ids_kb)
    learn = RNN_Learner(md, SingleModel(to_gpu(rnn)), opt_fn=opt_fn)
    learn.crit = seq2seq_loss

    learn.fit(config.lr, 1, cycle_len=12, use_clr=(20, 10))
    learn.save('initial')

    learn.load('initial')
    test(learn, val_dl, en_itos)

if __name__ == "__main__":
    main()