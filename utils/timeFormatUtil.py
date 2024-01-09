'''
时区问题很烦，这个类复杂处理项目可能涉及的时间格式
ps.我在数据库存时间的时候用的是SQL的DateTime，这个类型没有时区信息，库中的时间都是北京时间
'''
from datetime import datetime

import pytz
from dateutil.parser import parse


class timeFormatUtilClass:

    def utc2local(self, utcTimeStr) -> datetime:
        '''
        将utc时间转换为本地时间，
        前端如果用了JSON.stringify()，那么时间格式就是utc时间，接收这类时间的后端需要调用这个方法
        '''
        utcDateTime = parse(utcTimeStr)
        local_tz = pytz.timezone('Asia/Shanghai')
        localDateTime = utcDateTime.astimezone(local_tz)
        return localDateTime

    def addLocalTzinfo(self, nonDateTime: datetime):
        '''
        给一个没有时区信息的datetime加上北京时区信息，因为flask.jsonify()会把无时区信息的datetime视为utc时间，所以需要加上时区信息
        :param nonDateTime:
        :return:
        '''
        local_tz = pytz.timezone('Asia/Shanghai')
        local_tz.localize(nonDateTime)
        localDateTime = local_tz.localize(nonDateTime)
        return localDateTime

TIMEFORMAT = timeFormatUtilClass()
