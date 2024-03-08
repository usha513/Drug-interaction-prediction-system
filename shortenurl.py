pip install pyshorteners
pip install pyperclip
import pyshorteners
url=input("https://mail.google.com/mail/u/0/#inbox/KtbxLwgZXTMHCkzgHBZTqHzkjjHvWFlJtL")
def shorternurl(url):
    s=pyshorteners.Shortener()
    print(s.tinyurl.short(url))
shorternurl(url)