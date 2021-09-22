'''
Created on 26 Jul 2021

@author: matth
'''

import mysql.connector

def Conn():
    conn = mysql.connector.connect(host = 'localhost', user = 'root', password = 'Phpcampbell8', database = 'marmstrong21')
    return conn

#Example method to send SQL requests to database
#conn = cn.Conn()
#    cursor = conn.cursor()
#    cursor.execute("show tables")
#    for i in cursor:
#        st.write(i)





