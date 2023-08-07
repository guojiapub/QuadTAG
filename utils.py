import torch


def transform_doc_tokens_to_sent_tokens(hidden_states, sent_mask, max_sent_num, max_sent_len):
    bsz, _, embed_dim = hidden_states.size()
    sent_embeds = torch.zeros(bsz, max_sent_num, max_sent_len, embed_dim).to(hidden_states.device)
    sent_attn_masks = torch.zeros(bsz, max_sent_num, max_sent_len).to(hidden_states.device)
    for i in range(bsz):
        st = 0
        sent_id = 0
        for idx, ind in enumerate(sent_mask[i]):
            if ind == 1:
                sent_embeds[i][sent_id][:idx+1-st] = hidden_states[i][st:idx+1]
                sent_attn_masks[i][sent_id][:idx+1-st] = 1
                st = idx + 1
                sent_id += 1

    return sent_embeds, sent_attn_masks

