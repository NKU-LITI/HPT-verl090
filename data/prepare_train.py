from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

# print(dataset[0])

ret_dict = []
for item in dataset:
    # 删掉system prompt
    item['prompt'] = item['prompt'][1:]
    ret_dict.append(item)

print(ret_dict[0])

train_df = pd.DataFrame(ret_dict)
train_df.to_parquet("openr1.parquet")