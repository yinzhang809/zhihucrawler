from zhihu_oauth import ZhihuClient
import json
import random

client = ZhihuClient()
client.load_token('tokem.pkl')
convCont=0
colle=client.collection(19928423)

for answer in colle.answers:
    random_num=random.randint(0,50)
    if not (random_num==17):
        continue
    res = []
    comments = answer.comments
    if convCont>200000:
        break
    for comm in comments:
        try:
            conver = comm.replies
            a = comm.content
            i = 0
            for item in conver:
                if i == 0:
                    j=0
                else:
                    comItem={}
                    comItem["id"]=convCont
                    comItem["post"]=a.replace("<p>","").replace("</p>","")
                    comItem["res"]=item.content.replace("<p>","").replace("</p>","")
                    res.append(comItem)
                    convCont+=1
                i = i + 1
        except:
            continue
    with open("./zhihu_1.json", 'a', encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


# print(comm.content)
# print(answer.comments)
