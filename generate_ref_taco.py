import torch
from models.fatchord_version import WaveRNN
import hparams as hp
from utils.text.symbols import symbols
from models.tacotron_ref_encoder import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
import zipfile, os
from utils.paths import Paths
from utils.dataset import get_eval_dataset














if __name__ == "__main__" :

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation (lower quality)')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slower Unbatched Generation (better quality)')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(target=hp.voc_target)
    parser.set_defaults(overlap=hp.voc_overlap)
    parser.set_defaults(input_text=None)
    parser.set_defaults(weights_path=None)
    args = parser.parse_args()

    batched = args.batched
    target = args.target
    overlap = args.overlap
    input_text = args.input_text
    weights_path = args.weights_path


    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode='MOL').cuda()

    voc_model.restore('checkpoints/ljspeech_final_vocoder.wavernn/wave_rnn_600k_steps.pyt')

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout,
                         ref_enc_filters=hp.ref_enc_filters).cuda()


    tts_model.restore('checkpoints/lj_reference_encoder.tacotron/checkpoint_200k_steps.pyt')
    
    if input_text :
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else :
        with open('test_sentences_all.txt') as f :
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    r = tts_model.get_r()

    simple_table([('WaveRNN', str(voc_k) + 'k'),
                  (f'Tacotron(r={r})', str(tts_k) + 'k'),
                  ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    eval_set, _ = get_eval_dataset(paths.data, 1,5)
    ref_m = None

    for j, (x, m, ids, _) in enumerate(eval_set, 1):
        ref_m = m.cuda()
        print("Generating with id: ", ids)

        for i, x in enumerate(inputs, 1):

                print(f'\n| Generating {i}/{len(inputs)}')
                _, m, attention = tts_model.generate(x, ref_m)

                if input_text :
                    save_path = f'quick_start/__input_{input_text[:10]}_{tts_k}k.wav'
                else :
                    save_path = f'model_outputs/ref_taco/{i}_transfer_{ids}_{tts_k}k.wav'

                save_attention(attention, save_path)

                m = torch.tensor(m).unsqueeze(0)
                m = (m + 4) / 8

                voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

    print('\n\nDone.\n')
