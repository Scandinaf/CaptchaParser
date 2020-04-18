import calendar
import time

import ImageParser as imageParser

for _ in range(0, 1):
    timestamp = calendar.timegm(time.gmtime())
    imageParser.collect_images_for_model('https://www.mvd.gov.by/api/captcha/main?unique=$timestamp',
                                         "./test/validation")
    time.sleep(1)
