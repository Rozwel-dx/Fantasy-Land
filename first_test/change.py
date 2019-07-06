import xlrd
import pymysql

def open_excel():
    try:
        book = xlrd.open_workbook("kami3.xlsx")  # 文件名，把文件与py文件放在同一目录下
    except:
        print("open excel file failed!")
    try:
        sheet = book.sheet_by_name("Sheet1")  # execl里面的worksheet1
        return sheet
    except:
        print("locate worksheet in excel failed!")


# 连接数据库
try:
    db = pymysql.connect(host='localhost', user='eve', passwd='RfbYBNtFSvDXtejW', db='wx_group',charset='utf8')
except:
    print("could not connect to mysql server")


def search_count():
    cursor = db.cursor()
    select = "select count(id) from group_message"  # 获取表中记录数
    cursor.execute(select)  # 执行sql语句
    line_count = cursor.fetchone()
    print(line_count[0])


def insert_deta():
    sheet = open_excel()
    cursor = db.cursor()
    for i in range(1, sheet.nrows):  # 第一行是标题名，对应表中的字段名所以应该从第二行开始，计算机以0开始计数，所以值是1

        group_name = sheet.cell(i, 0).value
        group_intro = sheet.cell(i, 1).value
        group_label = sheet.cell(i, 2).value
        group_area = sheet.cell(i, 3).value
        group_tmt = sheet.cell(i, 4).value
        group_addtime = sheet.cell(i, 5).value
        group_img = sheet.cell(i, 6).value
        value = (group_name,group_intro,group_tmt,group_area,group_addtime,group_label,group_img)
        sql = "INSERT INTO group_message(group_name,group_intro,group_tmt,group_area,group_addtime,group_label,group_img)VALUES(%s,%s,%s,%s,%s,%s,%s)"
        cursor.execute(sql, value)  # 执行sql语句
        db.commit()
    cursor.close()  # 关闭连接


insert_deta()

db.close()  # 关闭数据
print("ok ")