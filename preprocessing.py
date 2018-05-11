import json
from functools import reduce

def combine():
    """
    since the crawler may break down because of Internet Error,
    I run it several times, and need to combine then into a single file.
    the file is named as zhihu_X.json. X is run id
    :return: combined list. A item is a dict with three fields: id, post, res
    """
    # read file
    dialog = []
    for i in range(1, 10):
        with open("./zhihu_{}.json".format(i), 'r') as load_f:
            load_list = json.load(load_f)
            for list in load_list:
                dialog += list
    return dialog

def filter_and_clean(dialog):
    """
    remove duplicate and too long or too short dialog
    :return: a list of dialog
    """
    # remove too long or too short dialog.
    print("Before preprocessing, there are {} dialogs.".format(len(dialog)))
    for d in dialog:
        if (len(d['post'])<5) or (len(d['post'])>50):
            dialog.remove(d)
            continue
        if (len(d['res'])<5) or (len(d['res'])>50):
            dialog.remove(d)
            continue
    print("After remove too long or too short dialog, there are {} dialogs.".format(len(dialog)))

    # clean (remove </br>tags)
    for idx,item in enumerate(dialog):
        dialog[idx]['post']=dialog[idx]['post'].replace('<br>','').replace('</br>','')
        dialog[idx]['res']=dialog[idx]['res'].replace('<br>','').replace('</br>','')
        dialog[idx].pop('id')



    # new_dialog=[]
    # [new_dialog.append((i)) for i in dialog if not i in new_dialog]
    # remove duplicate pairs.
    f= lambda x,y:x if y in x else x+[y]
    new_dialog=reduce(f,[[],]+dialog)
    # dialog_set=set(dialog)
    # new_dialog=[]
    # for dia in dialog:
    #     flag=False
    #     for newD in new_dialog:
    #         if  ((dia['post']==newD['post']) and (dia['res']==newD['res'])):
    #             flag=True
    #     if not (flag):
    #         new_dialog.append(dia)
    print("After remove duplicate dialogs, there are {} dialogs.".format(len(new_dialog)))
    with open("./data/zhihu_processed.json", 'w', encoding="utf-8") as f:
        json.dump(new_dialog, f, ensure_ascii=False, indent=4)
    return new_dialog


if __name__ == '__main__':
    filter_and_clean(combine())
