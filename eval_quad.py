import json
model_name = ''
data_folder = 'QAM'
data_file = 'dev'

if data_file == 'test':
    mode = ''
else:
    mode = 'dev'

with open(f"./results/{mode}[gold]{model_name}.txt") as f:
    gen_golds = json.load(f)
with open(f"./results/{mode}[pred]{model_name}.txt") as f:
    gen_preds = json.load(f)

assert len(gen_golds) == len(gen_preds)

print(model_name, ",", data_folder)

stance_map = {'1': "supports the topic", "-1": "is against the topic"}
label_map = {'Research': 1, 'Expert': 2, 'Case': 3, 'Explanation': 4, 'Others':5, '1': 6, "-1": 7}

def extract(text, doc_id):
    res = []
    for item in text.split(" [SEP] "):  
        try:
            claim_stance, evi_type = item.split(" : ")
        except:
            continue

        claim_id, stance = claim_stance.split(" ")[0], " ".join(claim_stance.split(" ")[1:])
        evi_w_typs =  [evis.split(" ") for evis in evi_type.split(" | ") if len(evis.split(" ")) == 2]
        res.extend([[claim_id, stance, evi, typ, doc_id] for evi, typ in evi_w_typs])     
    return res

def parse_text(gen_outputs):
    y_preds = []
    for i in range(len(gen_outputs)):
        y_preds.extend(extract(gen_outputs[i], f'doc_{i}'))
    return y_preds

def compute_f1_score(preds, golds):
    correct = len(preds & golds)
    prec = correct * 1.0 / len(preds)
    recall = correct * 1.0 / len(golds)
    print("num pred:", len(preds), "num gold:", len(golds), 'num correct:', correct)
    f1 = 2 * prec * recall / (prec + recall)
    print(f'prec: {prec:4f}, recall: {recall:.4f}, f1: {f1:4f}')



golds = set(map(tuple, parse_text(gen_golds)))
preds = set(map(tuple, parse_text(gen_preds)))
print(list(golds)[:5])
print('========================')
print(list(preds)[:5])
print('========================')
print("   Model scores  ")
compute_f1_score(preds, golds)
print('========================')




