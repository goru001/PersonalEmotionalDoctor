from fastai.text import *

from config import config

def create_emb(vecs, itos, em_sz):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = torch.from_numpy(vecs[w]*3)
        except: miss.append(w)
    print(len(miss),miss[5:10])
    return emb

def rand_t(*sz): return torch.randn(sz)/math.sqrt(sz[0])
def rand_p(*sz): return nn.Parameter(rand_t(*sz))

class Seq2SeqRNN(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()
        self.nl, self.nh, self.out_sl = nl, nh, out_sl
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)

        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

    def forward(self, inp):
        sl, bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = self.out_enc(h)

        dec_inp = V(torch.zeros(bs).long())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp == 1).all(): break
        return torch.stack(res)

    def initHidden(self, bs):
        return V(torch.zeros(self.nl, bs, self.nh))


class Seq2SeqAttnRNN(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2, knowledge_base_ids=None):
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl, self.nh, self.out_sl = nl, nh, out_sl
        # Knowledge base - Only using first for now, because of computational constraints
        self.kb_ids = np.array(np.concatenate(knowledge_base_ids[:1]).ravel()).reshape((-1, 1))
        # Encoder for questions
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        # Encoder for knowledge base
        self.gru_enc_kb = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data

        self.W1 = rand_p(nh, em_sz_dec)
        self.l2 = nn.Linear(em_sz_dec, em_sz_dec)
        self.l3 = nn.Linear(em_sz_dec + nh + nh + em_sz_dec, em_sz_dec)
        self.V = rand_p(em_sz_dec)

        self.W1_kb = rand_p(nh, em_sz_dec)
        self.l2_kb = nn.Linear(em_sz_dec, em_sz_dec)
        self.V_kb = rand_p(em_sz_dec)
        # to store state from previous question-answer
        self.prev = None

    def forward(self, inp, y=None, ret_attn=False):
        sl, bs = inp.size()
        # generate initial hidden state: all zeros
        h = self.initHidden(bs)
        # pass thr input through embedding matrix
        emb = self.emb_enc_drop(self.emb_enc(inp))
        # pass embedding output and initial hidden state through GRU
        enc_out, h = self.gru_enc(emb, h)
        h = self.out_enc(h)

        h_kb = self.initHidden(1)
        # pass the knowledge base through embedding layer
        # Weight sharing between embedding layer of input and embedding layer of knowledge base
        emb_kb = self.emb_enc_drop(self.emb_enc(V(torch.from_numpy(self.kb_ids), False)))
        # pass embedding output and initial hidden state through the encoder
        enc_out_kb, h_kb = self.gru_enc_kb(emb_kb, h_kb)
        # Weight sharing again
        h_kb = self.out_enc(h_kb)

        dec_inp = V(torch.zeros(bs).long())
        if self.prev is None:
            # flow will come here for the very first batch
            self.prev = V(torch.zeros(bs).long())
        res, attns, attns_kb = [], [], []

        # First input for "attention over question" and "attention over knowledge base" - encoder output
        # Weight sharing is being done, mainly because we don't have enough conversational data as of yet TODO: Try without weight sharing when enough conversational data is available
        # "attention over question" depends upon : What was the question (w1e), What's the present state of decoder (w2h)
        # "attention over knowledge base" depends upon : What was the question (w1e), What's the present state of decoder and knowledge base (w1e_kb)
        # Increase dimension of knowledge base's encoder hidden state, because intuitively, we would need more memory there
        w1e = enc_out @ self.W1
        w1e_kb = enc_out_kb @ self.W1_kb
        for i in range(self.out_sl):
            # grabbing the hidden state of last layer (We have 2 GRU layers)
            # second input for "attention over question" - previous state of decoder
            w2h = self.l2(h[-1])
            w2h_kb = self.l2_kb(h[-1])
            u = F.tanh(w1e + w2h)
            u_kb = F.tanh(torch.cat([w1e_kb,w1e], 0) + w2h_kb)
            a = F.softmax(u @ self.V, 0)
            a_kb = F.softmax(u_kb @ self.V_kb, 0)
            attns.append(a)
            attns_kb.append(a_kb)
            Xa = (a.unsqueeze(2) * enc_out).sum(0)
            Xa_kb = (a_kb.unsqueeze(2) * torch.cat([enc_out_kb,enc_out], 0)).sum(0)
            # First thing
            emb = self.emb_dec(dec_inp)
            emb_prev_ans = self.emb_dec(self.prev)
            # Second thing
            wgt_enc = self.l3(torch.cat([emb, Xa, Xa_kb, emb_prev_ans], 1))
            outp, h = self.gru_dec(wgt_enc.unsqueeze(0), h)
            # Third thing
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp == 1).all(): break
            if (y is not None) and (random.random() < self.pr_force):
                if i >= len(y): break
                dec_inp = y[i]
        self.prev = dec_inp

        res = torch.stack(res)
        if ret_attn: res = res, torch.stack(attns)
        return res

    def initHidden(self, bs):
        return V(torch.zeros(self.nl, bs, self.nh))