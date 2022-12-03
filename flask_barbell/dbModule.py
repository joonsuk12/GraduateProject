import pymysql


class Database():
    def __init__(self):
        # mysql connection 연결
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            password='4214',
            db='letmeseeDB',
            charset='utf8'
        )
        # Dictionary cursor 생성
        self.cur = self.conn.cursor(pymysql.cursors.DictCursor)  # 다른 형태의 커서

    def execute(self, query, args={}):
        # SQL문 실행
        self.cur.execute(query, args)  # "CREATE TABLE ")

    def commit(self):
        self.conn.commit()
