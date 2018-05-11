import json
import jieba
import pandas as pd


def combine(data_path,label_path):
    result=[]
    data_file = open(data_path)

    data=json.load(data_file)
    label=pd.read_csv(label_path,header=None).values
    i=0
    for pair in data:
        conv=[]
        post=[]
        res=[]

        post.append(" ".join(jieba.cut(pair['post'])))
        post.append(label[i][0])
        i=i+1
        res.append(" ".join(jieba.cut(pair['res'])))
        res.append(label[i][0])

        conv.append(post)
        conv.append(res)
        result.append(conv)
        i=i+1
    with open("./data/zhihu_labeled.txt", 'w', encoding="utf-8") as f:
        f.writelines(str(result))
        # json.dump(result, f, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    combine("./data/zhihu_processed.json","./data/test_label.csv")

    label = pd.read_csv("./data/test_label.csv", header=None).values
    for i in range(0,6):
        print(i)
        y=label==i
        z=label[y]
        print(z.size)
