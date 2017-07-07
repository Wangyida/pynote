#coding:utf-8
from selenium import webdriver
#下面填入京东的用户名以及密码
jd_up={"ue":"13031173311","pd":"a1234567"}

chrome=webdriver.Chrome()
chrome.get(url="http://item.mi.com/product/10000041.html")
chrome.find_element_by_xpath(".//*[@id='J_userInfo']/a[1]").click()
chrome.find_element_by_xpath(".//*[@id='username']").clear()
chrome.find_element_by_xpath(".//*[@id='username']").send_keys('13031173311')
chrome.find_element_by_xpath(".//*[@id='pwd']").clear()
chrome.find_element_by_xpath(".//*[@id='pwd']").send_keys('a1234567')
chrome.find_element_by_id('login-button').click()

while True:
    try:
        chrome.get(url="http://item.mi.com/product/10000041.html")
        chrome.find_element_by_id("btn btn-primary btn-biglarge J_proBuyBtn add").click()
        print("抢购成功并且已下单！！")
        chrome.quit()
    except Exception:
        print("还未开始抢购！！")

