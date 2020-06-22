from tqdm import tqdm
import json

from options import get_inference_config
from models import XRef
import utils

def infer():
    parser = get_inference_config()
    args = parser.parse_args()

    ckpt_path = utils.get_latest_ckpt_path(args.ckpt_dir)
    print(f'Evaluating on {ckpt_path}')

    model = XRef(args)
    model.load_from_checkpoint(ckpt_path)

    # model.freeze()
    model.cuda()

    fout = open('output/' + args.output_path, 'w')
    results = dict() # mapping mentions to list of prediction results over all candidates

    neg_count, pos_count = 0, 0
    for batch in tqdm(model.test_dataloader()):
        net_input = utils.move_to_cuda(batch)
        _, output_probs, accuracy = model(net_input)
        output_probs = (output_probs[0] > 0.5).long().tolist()
        for ix, ins_id in enumerate(batch['id']):
            art_id, cmt_id, ment_id, cand_id = ins_id.split('_')
            cmt_text = batch['comment_text'][ix]
            cand_text = batch['cand_text'][ix]
            ment = batch['mention_tuple'][ix]
            label = batch['labels'][ix].item()

            ment_id = f'{art_id}_{cmt_id}_{ment_id}'
            if ment_id not in results:
                results[ment_id] = {'comment': cmt_text,
                                    'mention': ment,
                                    'candidates': []}

            results[ment_id]['candidates'].append((cand_text, output_probs[ix], int(label)))
            if output_probs[ix] == 1:
                pos_count += 1
            else:
                neg_count += 1
    for ment, rst in results.items():
        modified_output_obj = rst
        modified_output_obj['candidates'] = sorted(rst['candidates'], key=lambda x: x[-1], reverse=True)
        fout.write(json.dumps(modified_output_obj) + '\n')
    fout.close()
    print(pos_count)
    print(neg_count)

if __name__=='__main__':
    infer()