import requests
import xlsxwriter
from lxml import etree

def get_int_after(s,f):
    S = s.upper()
    F = f.upper()
    par = S.partition(F)
    int_str = ""
    for c in par[2]:
        if c in ("-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            int_str += c
        else:
            if c == ":" or c == "=" or c == " ":
                if int_str == "":
                    continue
            break
    return int(int_str)


def get_id(i,j):
    url = "https://www.weixinqun.com/group?p=" + str(j)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.89 Safari/537.36'}
    data = requests.get(url, headers=headers).text
    s = etree.HTML(data)
    group_head = s.xpath('//*[@id="tab_head"]/li[' + str(i) + ']/div/a')[0]
    group_id = etree.tostring(group_head, method='html')
    gid = get_int_after(str(group_id), "id=")
    return gid

def get_message(k,val):
    temp = 1
    name = 0
    img = 0
    intro = 0
    tmt = 0
    label = 0
    area = 0
    addtime = 0
    group_name = ''
    group_img = ''
    group_intro = ''
    group_label = ''
    group_area = ''
    group_tmt = ''
    group_addtime = ''
    url = "https://www.weixinqun.com/group?id=" + str(k)
    headers = {
        'user-Agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"}
    data = requests.get(url, headers=headers).text

    for line in data:
        if temp >= data.index('<title>') + 8 and temp <= data.index('</title>'):
            group_name = group_name + line
        if temp >= data.index('<span class="shiftcode"><img src="') + 35:
            if img == 0 and line != '"':
                group_img = group_img + line
            else:
                img = 1
        if temp >= data.index('简介：') + 229:
            if intro == 0 and line != '<':
                if line != " " and line != "\n":
                    group_intro = group_intro + line
            else:
                intro = 1
        if temp >= data.index('行业：') + 91:
            if tmt == 0 and line != '<':
                if line != " " and line != "\n":
                    group_tmt = group_tmt + line
            else:
                tmt = 1
        if temp >= data.index('地区：') + 96:
            if area == 0 and line != '<':
                if line != " " and line != "\n":
                    group_area = group_area + line
            else:
                area = 1
        if temp >= data.index('时间：') + 5:
            if addtime == 0 and line != '<':
                if line != " " and line != "\n":
                    group_addtime = group_addtime + line
            else:
                addtime = 1
        if temp >= data.index('<a href="">') + 13:
            if label == 0 and line != '<':
                if line != " " and line != "\n":
                    group_label = group_label + line
            else:
                label = 1
        temp = temp + 1
    print("群名：", group_name)
    print("二维码：", group_img)
    print("简介：", group_intro)
    print("行业：", group_tmt)
    print("地区：", group_area)
    print("添加时间：", group_addtime)
    print("标签：", group_label)
    data = [group_name,group_intro,group_tmt,group_area,group_addtime,group_label,group_img]
    return data

val = 2
workbook = xlsxwriter.Workbook('d:\kami3.xlsx')  #创建一个Excel文件
worksheet = workbook.add_worksheet()               #创建一个sheet
title = [U'微信群名称',U'简介',U'行业',U'地区',U'添加时间',U'标签',U'二维码']     #表格title
worksheet.write_row('A1',title)                    #title 写入Excel
for page in range(0,2):                           #手动修改获取页数，信息全部获取完毕后写入excel，共3252页
    for num in range(1,43):
        num0 = str(val)
        row = 'A' + num0
        id = get_id(num,page)                      #获取各个微信群网页的id
        if id:
            data = get_message(id,val)                        #获取各个微信群的信息
            val = val + 1
            worksheet.write_row(row, data)
workbook.close()




