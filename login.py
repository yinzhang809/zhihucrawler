from zhihu_oauth import ZhihuClient
from zhihu_oauth.exception import NeedCaptchaException

client=ZhihuClient()
try:
    client.login('account','pwd')
except NeedCaptchaException:
    with open('a.gif','wb') as f:
        f.write(client.get_captcha())
    captcha=input('please input captcha:')
    client.login('account','pwd',captcha)
    print("Exception")
client.save_token('token.pkl')