from transformed_transformer.utils import make_text_seq2seq_lm_data


def test_make_text_seq2seq_lm_data_shapes():
    src, dec, tgt, vocab = make_text_seq2seq_lm_data(n_samples=4, src_len=8, tgt_len=8, seed=0)
    assert src.shape == (4, 8)
    assert dec.shape == (4, 8)
    assert tgt.shape == (4, 8)
    assert vocab.size > 10
